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
   /* Go through all the potential trajectories and draw them */
   for(size_t i = 0; i < m_tPotentialTrajectories[&(c_entity)].size(); ++i)
   {
      DrawWaypoints(m_tPotentialTrajectories[&c_entity][i].Waypoints, CColor::GREEN);
   }
   /* Go through all the saved trajectories and draw them */
   for(size_t i = 0; i < m_tSavedTrajectories[&c_entity].size(); ++i)
   {
      DrawWaypoints(m_tSavedTrajectories[&c_entity][i].Waypoints, CColor::RED);
   }
}

/****************************************/
/****************************************/

void CTrajectoryCollectionQTUserFunctions::DrawWaypoints(const std::vector<CVector3>& c_waypoints,
   const CColor& c_color) {
   /* Start drawing segments when you have at least two points */
   if(c_waypoints.size() > 1) {
      size_t unStart = 0;
      size_t unEnd = 1;
      while(unEnd < c_waypoints.size()) {
         DrawRay(CRay3(c_waypoints[unEnd],
                       c_waypoints[unStart]), c_color);
         ++unStart;
         ++unEnd;
      }
   }
}

/****************************************/
/****************************************/

REGISTER_QTOPENGL_USER_FUNCTIONS(CTrajectoryCollectionQTUserFunctions, "trajectory_collection_qtuser_functions")
