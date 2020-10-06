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
      pcKh->GetRABEquippedEntity().SetRange(2.0f);
      m_pcRABSensors.push_back(cController.GetSensor<CCI_RangeAndBearingSensor>("range_and_bearing"));

      /* Create a trajectory vector */
      m_tPotentialTrajectories[pcKh] = std::vector<STrajectoryData>();
      m_tSavedTrajectories[pcKh] = std::vector<STrajectoryData>();
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

         auto itStarted = std::find_if(m_tPotentialTrajectories[pcKh].begin(), m_tPotentialTrajectories[pcKh].end(),
         [unNeighborId] (const STrajectoryData& s_traj) {return s_traj.TrackedRobot == unNeighborId;});

         /* If potential trajectory started */
         if(itStarted != m_tPotentialTrajectories[pcKh].end())
         {
            /* If discontinuous, clear potential trajectory */
            if(itStarted->PrevTime != m_unClock - 1)
            {
               m_tPotentialTrajectories[pcKh].erase(itStarted);
               continue;
            }
            /* If finished, save and remove from potential */
            else if(itStarted->PrevTime - itStarted->StartTime == SEQ_LENGTH - 1)
            {
               m_tSavedTrajectories[pcKh].push_back(*itStarted);
               m_tPotentialTrajectories[pcKh].erase(itStarted);
            }
            /* If continuous, continue in potential */
            else
            {
               /* Compute the rotation matrix between start and current
               robot reference frame */
               CQuaternion cWR(pcKh->GetEmbodiedEntity().GetOriginAnchor().Orientation);
               CQuaternion cWO(itStarted->StartOrientation);
               // CQuaternion cOR = cWO.Conjugate() * cWR;
               // CRotationMatrix3 cTransition(cOR);
               CRadians cXAngle, cYAngle, cZWOAngle, cZWRAngle;
               cWO.ToEulerAngles(cZWOAngle, cYAngle, cXAngle);
               cWR.ToEulerAngles(cZWRAngle, cYAngle, cXAngle);
               /* Compute the offset between current and start frame*/
               CVector3 cPOR = pcKh->GetEmbodiedEntity().GetOriginAnchor().Position - itStarted->StartPosition;
               /* Get vector to point of interest in current frame */
               CVector3 cPR = CVector3(tMsgs[j].Range*0.01, CRadians::PI_OVER_TWO - tMsgs[j].VerticalBearing, tMsgs[j].HorizontalBearing);
               /* Get vector to point of interest in start frame */
               // CVector3 cPO = cTransition * cPR + cPOR;
               CVector3 cPW;
               cPW = (CVector3(cPR)).RotateZ(cZWRAngle) + pcKh->GetEmbodiedEntity().GetOriginAnchor().Position;
               // CVector3 cPO = cPW.RotateZ(cZWOAngle) + itStarted->StartPosition;
               CVector3 cPO = (cPW - itStarted->StartPosition).RotateZ(-cZWOAngle);
               /* Add this point to waypoints */
               itStarted->Waypoints.push_back(cPO);
               itStarted->PrevTime = m_unClock;

            }
         }
         /* Else, start a potential trajectory */
         else if (unNeighborId != 0)
         {
            STrajectoryData sNewTraj;
            sNewTraj.StartTime = m_unClock;
            sNewTraj.PrevTime = m_unClock;
            sNewTraj.TrackedRobot = unNeighborId;
            sNewTraj.StartOrientation = pcKh->GetEmbodiedEntity().GetOriginAnchor().Orientation;
            sNewTraj.StartPosition = pcKh->GetEmbodiedEntity().GetOriginAnchor().Position;
            
            CVector3 cPO = CVector3(tMsgs[j].Range*0.01, CRadians::PI_OVER_TWO - tMsgs[j].VerticalBearing, tMsgs[j].HorizontalBearing);
            sNewTraj.Waypoints.push_back(cPO);
            m_tPotentialTrajectories[pcKh].push_back(sNewTraj);
         }   
      }

      /* Flush every 10 saved trajectories for the robot */
      if(m_tSavedTrajectories[pcKh].size() >= 10)
      {
         std::string strBuffer;
         UInt32 unCountTime = 0;
         for (auto itTrajectory = m_tSavedTrajectories[pcKh].begin();
               itTrajectory != m_tSavedTrajectories[pcKh].end();
               ++itTrajectory)
         {
            for(auto itPoint = itTrajectory->Waypoints.begin();
                itPoint != itTrajectory->Waypoints.end();
                ++itPoint)
            {
               strBuffer += pcKh->GetId().substr(2) + ", "
                           + ToString(itTrajectory->TrackedRobot) + ", " 
                           + ToString(itTrajectory->StartTime + unCountTime) + ", "
                           + ToString(itPoint->GetX()) + ", "
                           + ToString(itPoint->GetY()) + ", "
                           + ToString(0) + ", "
                           + "\n";
               ++unCountTime;
            }
            strBuffer += "\n";
         }
         m_cOutput << strBuffer;
         m_tSavedTrajectories[pcKh].clear();

         /* Flush unfinished potential trajectories at the same time */
         for (auto itTrajectory = m_tPotentialTrajectories[pcKh].begin();
               itTrajectory != m_tPotentialTrajectories[pcKh].end();
               )
         {
            if(m_unClock - itTrajectory->PrevTime > 2)
               itTrajectory = m_tPotentialTrajectories[pcKh].erase(itTrajectory);
            else ++itTrajectory;
         }

      }
   }
   m_unClock++;

}

/****************************************/
/****************************************/

void CTrajectoryCollectionLoopFunctions::PostExperiment() {
   std::string strBuffer;
   for(auto itElement = m_tSavedTrajectories.begin(); 
      itElement != m_tSavedTrajectories.end(); itElement++)
   {
      std::string strRobotId = itElement->first->GetId().substr(2);
      std::vector<STrajectoryData> vecTrajectories = itElement->second;

      for(auto itTrajectory = vecTrajectories.begin();
          itTrajectory != vecTrajectories.end(); ++itTrajectory)
      {
         UInt32 unCountTime = 0;
         for (auto itPoint = (itTrajectory->Waypoints).begin();
            itPoint != (itTrajectory->Waypoints).end(); ++itPoint)
         {
           strBuffer += strRobotId + ", "
                        + ToString(itTrajectory->TrackedRobot) + ", " 
                        + ToString(itTrajectory->StartTime + unCountTime) + ", "
                        + ToString(itPoint->GetX()) + ", "
                        + ToString(itPoint->GetY()) + ", "
                        + ToString(0) + ", "
                        + "\n";
           ++unCountTime;
         }
         strBuffer += "\n";
      }
   }
   if (!strBuffer.empty())
      m_cOutput << strBuffer;
   m_cOutput.close();
}

REGISTER_LOOP_FUNCTIONS(CTrajectoryCollectionLoopFunctions, "trajectory_collection_loop_functions")
