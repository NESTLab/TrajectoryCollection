#include "trajectory_collection_qtuser_functions.h"
#include "trajectory_collection_loop_functions.h"

/****************************************/
/****************************************/

CTrajectoryCollectionQTUserFunctions::CTrajectoryCollectionQTUserFunctions() :
   m_cTrajLF(dynamic_cast<CTrajectoryCollectionLoopFunctions&>(CSimulator::GetInstance().GetLoopFunctions()))
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
}

/****************************************/
/****************************************/

void CTrajectoryCollectionQTUserFunctions::Draw(CKheperaIVEntity& c_entity) {
   /* Draw robot id */
   DrawText(CVector3(0.0, 0.0, 0.3),   // position
            c_entity.GetId().c_str()); // text

   CVector3 cCurrentPos = c_entity.GetEmbodiedEntity().GetOriginAnchor().Position;
   CQuaternion cCurrentOrientation = c_entity.GetEmbodiedEntity().GetOriginAnchor().Orientation;
   if(c_entity.GetId() == "kh1")
   {
      /* Go through all the potential trajectories and draw them */
      for(size_t i = 0; i < m_tPotentialTrajectories[&(c_entity)].size(); ++i)
      {
         DrawWaypoints(m_tPotentialTrajectories[&c_entity][i].Waypoints,
                       m_tPotentialTrajectories[&c_entity][i].StartPosition,
                       m_tPotentialTrajectories[&c_entity][i].StartOrientation,
                       cCurrentPos,
                       cCurrentOrientation,
                       CColor::GREEN);
      }
      /* Go through all the saved trajectories and draw them */
      for(size_t i = 0; i < m_tSavedTrajectories[&c_entity].size(); ++i)
      {
         DrawWaypoints(m_tSavedTrajectories[&c_entity][i].Waypoints,
                       m_tSavedTrajectories[&c_entity][i].StartPosition,
                       m_tSavedTrajectories[&c_entity][i].StartOrientation,
                       cCurrentPos,
                       cCurrentOrientation,
                       CColor::RED);
      }
   }
}

/****************************************/
/****************************************/

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
      CQuaternion cRO = c_WR.Inverse() * c_WO;
      CRotationMatrix3  cTransition(cRO);//c_startOrientation.Inverse() * c_currOrientation);
      cStart = cTransition * c_waypoints[unStart] + (c_startPosition - c_currPosition);
      while(unEnd < c_waypoints.size()) {
         cEnd = cTransition * c_waypoints[unEnd] + (c_startPosition - c_currPosition);
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
