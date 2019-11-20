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
static const UInt32 SEQ_LENGTH = 80;

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
      // m_pcPosSensors.push_back(cController.GetSensor<CCI_PositioningSensor>("positioning"));
      /* Create a trajectory vector */
      m_tPotentialTrajectories[pcKh] = TPerKhepTrajectoryMap();
      //std::make_pair<UInt32(0), std::vector<STrajectoryData>()>;
      m_tSavedTrajectories[pcKh] = TPerKhepTrajectoryMap();
      //std::make_pair<UInt32(0), std::vector<STrajectoryData>()>;
   }
   m_unClock = 0;
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
               CRotationMatrix3 cTransition(
                  (m_tPotentialTrajectories[pcKh][tMsgs[j].Data[0]].StartOrientation) *
                  (pcKh->GetEmbodiedEntity().GetOriginAnchor().Orientation).Inverse()
               );
               /* Get vector to point of interest in current frame */
               CVector3 cP1(tMsgs[j].Range, CRadians(0.0), tMsgs[j].HorizontalBearing);
               /* Get vector to point of interest in start frame */
               CVector3 cP0 = cTransition * cP1 
               + m_tPotentialTrajectories[pcKh][tMsgs[j].Data[0]].StartPosition;
               /* Add this point to waypoints */
               m_tPotentialTrajectories[pcKh][tMsgs[j].Data[0]].Waypoints.push_back(cP0);
               /* */
               m_tPotentialTrajectories[pcKh][tMsgs[j].Data[0]].PrevTime = m_unClock;
               
            }
         }
         /* Else, start a potential trajectory */
         else
         {
            STrajectoryData sNewTraj;
            sNewTraj.StartTime = m_unClock;
            sNewTraj.PrevTime = m_unClock;
            sNewTraj.StartOrientation = pcKh->GetEmbodiedEntity().GetOriginAnchor().Orientation; // ->GetEmbodiedEntity().GetOriginAnchor().Position
            sNewTraj.StartPosition = pcKh->GetEmbodiedEntity().GetOriginAnchor().Position;
            sNewTraj.Waypoints.push_back(CVector3(tMsgs[j].Range,
               CRadians(0.0), tMsgs[j].HorizontalBearing)); // to do: convert elevation to inclination for 3D
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
   // for(TTrajectoryMap::iterator it = m_tSavedTrajectories.begin(); 
   //    it != m_tSavedTrajectories.end(); it++)
   // {
   //    for(std::vector<char> v; it->second)
   //    strBuffer += ToString() + ", " 
   //              + "\n";
   // }
   // if (!strBuffer.empty())
   //    m_strDataset << strBuffer << std::endl;
}

REGISTER_LOOP_FUNCTIONS(CTrajectoryCollectionLoopFunctions, "trajectory_collection_loop_functions")
