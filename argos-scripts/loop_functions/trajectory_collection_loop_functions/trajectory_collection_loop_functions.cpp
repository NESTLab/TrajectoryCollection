#include "trajectory_collection_loop_functions.h"

/****************************************/
/****************************************/

/*
 * To reduce the number of waypoints stored in memory,
 * consider two robot positions distinct if they are
 * at least MIN_DISTANCE away from each other
 * This constant is expressed in meters
 */
static const Real MIN_DISTANCE = 0.05f;
/* Convenience constant to avoid calculating the square root in PostStep() */
static const Real MIN_DISTANCE_SQUARED = MIN_DISTANCE * MIN_DISTANCE;
/* Length of the sequences (in number of time steps) */
static const UInt32 SEQ_LENGTH = 100;

/****************************************/
/****************************************/

void CTrajectoryCollectionLoopFunctions::Init(TConfigurationNode& t_tree) {

   TConfigurationNode& tCollection = GetNode(t_tree, "collection");
   try {
      GetNodeAttribute(tCollection, "filename", m_strFilename);
   }
   catch(...)
   {
      m_strFilename = "dataset.dat";
   }

   /* 
    * Go through all the robots in the environment
    * and create an entry in the waypoint map for each of them
    */
   /* Get the map of all kheperas from the space */
   CSpace::TMapPerType& tKhMap = GetSpace().GetEntitiesByType("kheperaiv");
   /* Go through them */
   for(CSpace::TMapPerType::iterator it = tKhMap.begin();
       it != tKhMap.end();
       ++it) {
      /* Create a pointer to the current Khepera */
      CKheperaIVEntity* pcKh = any_cast<CKheperaIVEntity*>(it->second);
      m_pcKheperas.push_back(pcKh);
      CCI_Controller& cController = pcKh->GetControllableEntity().GetController();
      m_pcRABSensors.push_back(cController.GetSensor<CCI_RangeAndBearingSensor>("range_and_bearing"));
      /* Create a trajectory vector */
      m_tPotentialTrajectories[pcKh] = TPerKhepTrajectoryMap();
      m_tSavedTrajectories[pcKh] = TPerKhepTrajectoryMap();
   }
   m_unClock = 0;
   m_cOutput.open(m_strFilename , std::ios_base::trunc | std::ios_base::out);
   m_cGraphOutput.open("G_" +m_strFilename, std::ios_base::trunc | std::ios_base::out);
}

/****************************************/
/****************************************/

void CTrajectoryCollectionLoopFunctions::Reset() {
   /*
    * Clear all the waypoint vectors
    */

}

/****************************************/
/****************************************/

void CTrajectoryCollectionLoopFunctions::PostStep() {
   /* Go through Kheperas */
   CVector3 cPosRobot2;
   for(size_t i = 0; i < m_pcKheperas.size(); ++i) {
      CKheperaIVEntity* pcKh = m_pcKheperas[i];
      const CCI_RangeAndBearingSensor::TReadings& tMsgs = 
      m_pcRABSensors[i]->GetReadings();

      /* Go through current RAB readings for Khepera */
      for(size_t j = 0; j < tMsgs.size(); ++j)
      {
         UInt32 unNeighborId = tMsgs[j].Data[0];
         
         m_cGraphOutput << pcKh->GetId().substr(2) + ", " 
         + ToString(m_unClock) + ", " + ToString(unNeighborId) << std::endl;

         /* If potential trajectory started */
         if(m_tPotentialTrajectories[pcKh].count(
            unNeighborId) != 0)
         {
            /* If discontinuous, clear potential and start */
            if(m_tPotentialTrajectories[pcKh][unNeighborId].PrevTime 
               != m_unClock - 1)
            {
               m_tPotentialTrajectories[pcKh].erase(unNeighborId);
               continue;
            }
            /* If finished, save and remove from potential */
            else if(m_tPotentialTrajectories[pcKh][unNeighborId].PrevTime
               - m_tPotentialTrajectories[pcKh][unNeighborId].StartTime 
               == SEQ_LENGTH - 1)
            {
               m_tSavedTrajectories[pcKh][unNeighborId] = 
               m_tPotentialTrajectories[pcKh][unNeighborId];
               
               m_tPotentialTrajectories[pcKh].erase(unNeighborId);

               std::string strBuffer;
               UInt32 unCountTime = 0;
               for (auto it_traj = 
                  (m_tSavedTrajectories[pcKh][unNeighborId].Waypoints).begin();
                  it_traj != (m_tSavedTrajectories[pcKh][unNeighborId].Waypoints).end(); ++it_traj)
               {
                  strBuffer += pcKh->GetId().substr(2) + ", "
                              + ToString(unNeighborId) + ", " 
                              + ToString(m_tSavedTrajectories[pcKh][unNeighborId].StartTime
                                 + unCountTime) + ", "
                              + ToString(it_traj->GetX()) + ", "
                              + ToString(it_traj->GetY()) + ", "
                              + ToString(it_traj->GetZ()) + ", "
                              + "\n";
                  ++unCountTime;
               }
               m_cOutput << strBuffer << std::endl;
            }
            /* If continuous, continue in potential */
            else
            {
               /* Compute the rotation matrix between start and current
               robot reference frame */
               CQuaternion cWR(pcKh->GetEmbodiedEntity().GetOriginAnchor().Orientation);
               CQuaternion cWO(m_tPotentialTrajectories[pcKh][unNeighborId].StartOrientation);
               // CQuaternion cOR = cWO.Conjugate() * cWR;
               // CRotationMatrix3 cTransition(cOR);
               CRadians cXAngle, cYAngle, cZWOAngle, cZWRAngle;
               cWO.ToEulerAngles(cZWOAngle, cYAngle, cXAngle);
               cWR.ToEulerAngles(cZWRAngle, cYAngle, cXAngle);
               /* Compute the offset between current and start frame*/
               CVector3 cPOR = pcKh->GetEmbodiedEntity().GetOriginAnchor().Position - m_tPotentialTrajectories[pcKh][unNeighborId].StartPosition;
               /* Get vector to point of interest in current frame */
               CVector3 cPR = CVector3(tMsgs[j].Range*0.01, CRadians::PI_OVER_TWO - tMsgs[j].VerticalBearing, tMsgs[j].HorizontalBearing);
               /* Get vector to point of interest in start frame */
               // CVector3 cPO = cTransition * cPR + cPOR;
               CVector3 cPW;
               cPW = (CVector3(cPR)).RotateZ(cZWRAngle) + pcKh->GetEmbodiedEntity().GetOriginAnchor().Position;
               CVector3 cPO = cPW.RotateZ(cZWOAngle) + m_tPotentialTrajectories[pcKh][unNeighborId].StartPosition;
               /* Add this point to waypoints */
               m_tPotentialTrajectories[pcKh][unNeighborId].Waypoints.push_back(cPO);
               m_tPotentialTrajectories[pcKh][unNeighborId].PrevTime = m_unClock;

            }
         }
         /* Else, start a potential trajectory */
         else if (unNeighborId != 0)
         {
            STrajectoryData sNewTraj;
            sNewTraj.StartTime = m_unClock;
            sNewTraj.PrevTime = m_unClock;
            sNewTraj.StartOrientation = m_tSavedTrajectories[pcKh][unNeighborId].StartOrientation;
            sNewTraj.StartPosition = m_tSavedTrajectories[pcKh][unNeighborId].StartPosition;
            m_tPotentialTrajectories[pcKh][unNeighborId] = sNewTraj;
         }   
      }
   }
   m_unClock++;
}

/****************************************/
/****************************************/

void CTrajectoryCollectionLoopFunctions::PostExperiment() {
   std::string strBuffer;
   for(TTrajectoryMap::iterator it = m_tSavedTrajectories.begin(); 
      it != m_tSavedTrajectories.end(); it++)
   {
      for(auto it_tracked = it->second.begin();
          it_tracked != it->second.end(); ++it_tracked)
      {
         UInt32 unCountTime = 0;
         for (auto it_traj = (it_tracked->second.Waypoints).begin();
            it_traj != (it_tracked->second.Waypoints).end(); ++it_traj)
         {
           strBuffer += it->first->GetId().substr(2) + ", "
                        + ToString(it_tracked->first) + ", " 
                        + ToString((it_tracked->second).StartTime + unCountTime) + ", "
                        + ToString(it_traj->GetX()) + ", "
                        + ToString(it_traj->GetY()) + ", "
                        + ToString(it_traj->GetZ()) + ", "
                        + "\n";
           ++unCountTime;
         }
      }
   }
   if (!strBuffer.empty())
      m_cOutput << strBuffer << std::endl;
   m_cOutput.close();
}

REGISTER_LOOP_FUNCTIONS(CTrajectoryCollectionLoopFunctions, "trajectory_collection_loop_functions")
