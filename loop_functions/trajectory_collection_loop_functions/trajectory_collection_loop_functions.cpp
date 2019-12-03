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
   m_strFilename = "dataset.dat";
   m_cOutput.open(m_strFilename , std::ios_base::trunc | std::ios_base::out);
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
   CVector3 cPos2, cPos1;
   for(size_t i = 0; i < m_pcKheperas.size(); ++i) {
      CKheperaIVEntity* pcKh = m_pcKheperas[i];
      const CCI_RangeAndBearingSensor::TReadings& tMsgs = 
      m_pcRABSensors[i]->GetReadings();
     
      /* Go through current RAB readings for Khepera */
      for(size_t j = 0; j < tMsgs.size(); ++j)
      {
         /* If potential trajectory started */
         if(m_tPotentialTrajectories[pcKh].count(
            tMsgs[j].Data[0]) != 0)
         {
            /* If discontinuous, clear potential and start */
            if(m_tPotentialTrajectories[pcKh][tMsgs[j].Data[0]].PrevTime 
               != m_unClock - 1)
            {
               m_tPotentialTrajectories[pcKh].erase(tMsgs[j].Data[0]);
               continue;
            }
            /* If finished, save and remove from potential */
            else if(m_tPotentialTrajectories[pcKh][tMsgs[j].Data[0]].PrevTime
               - m_tPotentialTrajectories[pcKh][tMsgs[j].Data[0]].StartTime 
               == SEQ_LENGTH - 1)
            {
               m_tSavedTrajectories[pcKh][tMsgs[j].Data[0]] = 
               m_tPotentialTrajectories[pcKh][tMsgs[j].Data[0]];
               m_tPotentialTrajectories[pcKh].erase(tMsgs[j].Data[0]);
            }
            /* If continuous, continue in potential */
            else
            {
               /* Compute the rotation matrix between start and current
               robot reference frame */
               CQuaternion cWR(pcKh->GetEmbodiedEntity().GetOriginAnchor().Orientation);
               CQuaternion cWO(m_tPotentialTrajectories[pcKh][tMsgs[j].Data[0]].StartOrientation);

               // CQuaternion cOR = cWO.Conjugate() * cWR;
               // CRotationMatrix3 cTransition(cOR);

               CRadians cXAngle, cYAngle, cZWOAngle, cZWRAngle;
               cWO.ToEulerAngles(cZWOAngle, cYAngle, cXAngle);
               cWR.ToEulerAngles(cZWRAngle, cYAngle, cXAngle);
               /* Compute the offset between current and start frame*/
               CVector3 cPOR = pcKh->GetEmbodiedEntity().GetOriginAnchor().Position - m_tPotentialTrajectories[pcKh][tMsgs[j].Data[0]].StartPosition;
               /* Get vector to point of interest in current frame */
               CVector3 cPR(tMsgs[j].Range/100 * Cos(tMsgs[j].HorizontalBearing), tMsgs[j].Range/100 * Sin(tMsgs[j].HorizontalBearing), 0.0);
               /* Get vector to point of interest in start frame */
               // CVector3 cPO = cTransition * cPR + cPOR;
               CVector3 cPO = cPR.RotateZ(cZWRAngle-cZWOAngle) + cPOR;
               /* Add this point to waypoints */
               m_tPotentialTrajectories[pcKh][tMsgs[j].Data[0]].Waypoints.push_back(cPO);
               m_tPotentialTrajectories[pcKh][tMsgs[j].Data[0]].PrevTime = m_unClock;
            }
         }
         /* Else, start a potential trajectory */
         else
         {
            STrajectoryData sNewTraj;
            sNewTraj.StartTime = m_unClock;
            sNewTraj.PrevTime = m_unClock;
            sNewTraj.StartOrientation = pcKh->GetEmbodiedEntity().GetOriginAnchor().Orientation;
            sNewTraj.StartPosition = pcKh->GetEmbodiedEntity().GetOriginAnchor().Position;
            sNewTraj.Waypoints.push_back(CVector3(tMsgs[j].Range/100 * Cos(tMsgs[j].HorizontalBearing), tMsgs[j].Range/100 * Sin(tMsgs[j].HorizontalBearing), 0.0));
            m_tPotentialTrajectories[pcKh][tMsgs[j].Data[0]] = sNewTraj;
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
         for (auto it_traj = (it_tracked->second.Waypoints).begin();
            it_traj != (it_tracked->second.Waypoints).end(); ++it_traj)
         {
           strBuffer += it->first->GetId().substr(2) + ", "
                        + ToString(it_tracked->first) + ", " 
                        + ToString(it_traj->GetX()) + ", "
                        + ToString(it_traj->GetY()) + ", "
                        + ToString(it_traj->GetZ()) + ", "
                        + "\n";
         }
      }
   }
   if (!strBuffer.empty())
      m_cOutput << strBuffer << std::endl;
   m_cOutput.close();
}

REGISTER_LOOP_FUNCTIONS(CTrajectoryCollectionLoopFunctions, "trajectory_collection_loop_functions")
