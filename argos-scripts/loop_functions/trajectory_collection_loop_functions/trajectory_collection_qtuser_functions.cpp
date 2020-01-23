#include "trajectory_collection_qtuser_functions.h"
#include "trajectory_collection_loop_functions.h"
#include "../master_loop_functions/master_loop_functions.h"

/****************************************/
/****************************************/

CTrajectoryCollectionQTUserFunctions::CTrajectoryCollectionQTUserFunctions() :
   m_cTrajLF(dynamic_cast<CTrajectoryCollectionLoopFunctions&>((dynamic_cast<CMasterLoopFunctions&>(CSimulator::GetInstance().GetLoopFunctions())).GetLoopFunction("trajectory_collection_loop_functions")))
   {
      RegisterUserFunction<CTrajectoryCollectionQTUserFunctions, 
      CKheperaIVEntity>(&CTrajectoryCollectionQTUserFunctions::Draw);
   }


/****************************************/
/****************************************/

void CTrajectoryCollectionQTUserFunctions::DrawInWorld()
{
   m_tSavedTrajectories = m_cTrajLF.GetSavedTrajectories();
   m_tPotentialTrajectories = m_cTrajLF.GetPotentialTrajectories();

   for(auto element : m_tPotentialTrajectories)
   {
      CKheperaIVEntity& c_entity = *(element.first);
      if(c_entity.GetId() == "kh1") 
      {
         /* Go through all the potential trajectories and draw them */
         for(size_t i = 0; i < m_tPotentialTrajectories[&(c_entity)].size(); ++i)
         {
            DrawWaypointsInWorld(m_tPotentialTrajectories[&c_entity][i].Waypoints,
              m_tPotentialTrajectories[&c_entity][i].StartPosition,
              m_tPotentialTrajectories[&c_entity][i].StartOrientation,
              CColor::GREEN);
         }
      }
   }

   for(auto element : m_tSavedTrajectories)
   {
      CKheperaIVEntity& c_entity = *(element.first);
      if(c_entity.GetId() == "kh1") 
      {
         /* Go through all the saved trajectories and draw them */
         for(size_t i = 0; i < m_tSavedTrajectories[&c_entity].size(); ++i)
         {
            DrawWaypointsInWorld(m_tSavedTrajectories[&c_entity][i].Waypoints,
                          m_tSavedTrajectories[&c_entity][i].StartPosition,
                          m_tSavedTrajectories[&c_entity][i].StartOrientation,
                          CColor::RED);
         }      
      }
   }
}

/****************************************/
/****************************************/

void CTrajectoryCollectionQTUserFunctions::Draw(CKheperaIVEntity& c_entity) {
   /* Draw robot id */
   DrawText(CVector3(0.0, 0.0, 0.1),   // position
            c_entity.GetId().c_str()); // text

   // CVector3 cCurrentPos = c_entity.GetEmbodiedEntity().GetOriginAnchor().Position;
   // CQuaternion cCurrentOrientation = c_entity.GetEmbodiedEntity().GetOriginAnchor().Orientation;
   // if(c_entity.GetId() == "kh1")
   // {
   //    /* Go through all the potential trajectories and draw them */
   //    for(size_t i = 0; i < m_tPotentialTrajectories[&(c_entity)].size(); ++i)
   //    {
   //       DrawWaypoints(m_tPotentialTrajectories[&c_entity][i].Waypoints,
   //         m_tPotentialTrajectories[&c_entity][i].StartPosition,
   //         m_tPotentialTrajectories[&c_entity][i].StartOrientation,
   //         cCurrentPos,
   //         cCurrentOrientation,
   //         CColor::GREEN);
   //    }
   //    /* Go through all the saved trajectories and draw them */
   //    for(size_t i = 0; i < m_tSavedTrajectories[&c_entity].size(); ++i)
   //    {
   //       DrawWaypoints(m_tSavedTrajectories[&c_entity][i].Waypoints,
   //                     m_tSavedTrajectories[&c_entity][i].StartPosition,
   //                     m_tSavedTrajectories[&c_entity][i].StartOrientation,
   //                     cCurrentPos,
   //                     cCurrentOrientation,
   //                     CColor::RED);
   //    }
   // }
}

/****************************************/
/****************************************/

void CTrajectoryCollectionQTUserFunctions::DrawWaypointsInWorld(const std::vector<CVector3>& c_waypoints, 
   const CVector3& c_startPosition,
   const CQuaternion& c_WO,
   const CColor& c_color) {
   /* Start drawing segments when you have at least two points */
   if(c_waypoints.size() > 1) {
      size_t unStart = 0;
      size_t unEnd = 1;
      CVector3 cStart, cEnd;

      CRadians cXAngle, cYAngle, cZWOAngle;
      c_WO.ToEulerAngles(cZWOAngle, cYAngle, cXAngle);

      CVector3 cPWO = c_startPosition;
      CVector3 cTemp(c_waypoints[unStart]);
      cStart = (cTemp - cPWO).RotateZ(-cZWOAngle);
      while(unEnd < c_waypoints.size()) {
         cTemp = c_waypoints[unEnd];
         cEnd = (cTemp - cPWO).RotateZ(-cZWOAngle);
         DrawRay(CRay3(cEnd,
                       cStart), c_color);
         cStart = cEnd;
         ++unStart;
         ++unEnd;
      }
   }
}


void CTrajectoryCollectionQTUserFunctions::DrawWaypoints(const std::vector<CVector3>& c_waypoints, 
   const CVector3& c_startPosition,
   const CQuaternion& c_WO,
   const CVector3& c_currPosition,
   const CQuaternion& c_WR,
   const CColor& c_color) {
   /* Start drawing segments when you have at least two points */
   if(c_waypoints.size() > 1) {
      size_t unStart = 0;
      size_t unEnd = 1;
      CVector3 cStart, cEnd;

      CRadians cXAngle, cYAngle, cZWOAngle, cZWRAngle;
      c_WO.ToEulerAngles(cZWOAngle, cYAngle, cXAngle);
      c_WR.ToEulerAngles(cZWRAngle, cYAngle, cXAngle);

      CVector3 cPOR = c_currPosition - c_startPosition;
      // cStart = cTransition * c_waypoints[unStart] + cPRO;
      cStart = (CVector3(c_waypoints[unStart]) - cPOR).RotateZ(cZWOAngle - cZWRAngle);
      while(unEnd < c_waypoints.size()) {
         cEnd = (CVector3(c_waypoints[unEnd]) - cPOR).RotateZ(cZWOAngle - cZWRAngle);
         // cEnd = cTransition * c_waypoints[unEnd] + cPRO;
         DrawRay(CRay3(cEnd,
                       cStart), c_color);
         cStart = cEnd;
         ++unStart;
         ++unEnd;
      }
   }
}

/****************************************/
/****************************************/

REGISTER_QTOPENGL_USER_FUNCTIONS(CTrajectoryCollectionQTUserFunctions, "trajectory_collection_qtuser_functions")