#ifndef TRAJECTORY_QTUSER_FUNCTIONS_H
#define TRAJECTORY_QTUSER_FUNCTIONS_H

#include <argos3/plugins/simulator/visualizations/qt-opengl/qtopengl_user_functions.h>
#include <argos3/plugins/robots/kheperaiv/simulator/kheperaiv_entity.h>
#include "trajectory_collection_loop_functions.h"

using namespace argos;

class CTrajectoryCollectionLoopFunctions;

class CTrajectoryCollectionQTUserFunctions : public CQTOpenGLUserFunctions {

public:

   CTrajectoryCollectionQTUserFunctions();

   virtual ~CTrajectoryCollectionQTUserFunctions() {}

   virtual void DrawInWorld();

   void Draw(CKheperaIVEntity& c_entity);


private:

   void DrawWaypoints(const std::vector<CVector3>& c_waypoints,
   const CColor& c_color);

private:

   CTrajectoryCollectionLoopFunctions::TTrajectoryMap m_tPotentialTrajectories;
   CTrajectoryCollectionLoopFunctions::TTrajectoryMap m_tSavedTrajectories;
   CTrajectoryCollectionLoopFunctions& m_cTrajLF;

};

#endif
