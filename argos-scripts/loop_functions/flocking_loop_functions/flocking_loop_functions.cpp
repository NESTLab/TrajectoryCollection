#include "flocking_loop_functions.h"

static const CVector3 POS_1(2,1.5,1.0);
static const CVector3 POS_2(-2,-1.5,1.0);//8.9,2.7,0.5);

/****************************************/
/****************************************/

void CFlockingLoopFunctions::Init(TConfigurationNode& t_tree) {
   /*
    * Go through all the robots in the environment
    * and create an entry in the waypoint map for each of them
    */
   /* Get the map of all foot-bots from the space */
   m_pcLight = &dynamic_cast<CLightEntity&>(GetSpace().GetEntity("light"));
}

/****************************************/
/****************************************/

void CFlockingLoopFunctions::Reset() {
   MoveEntity(*m_pcPosLight,
               POS_2,
               CQuaternion());
}

/****************************************/
/****************************************/

void CFlockingLoopFunctions::PostStep() {

   if(GetSpace().GetSimulationClock()%2000 == 0)
   {
      m_pcPosLight = &dynamic_cast<CPositionalEntity&>(*m_pcLight);
      MoveEntity(*m_pcPosLight,
            POS_1,
            CQuaternion());
   }
   else if(GetSpace().GetSimulationClock()%1000 == 0)
   {
      m_pcPosLight = &dynamic_cast<CPositionalEntity&>(*m_pcLight);
      MoveEntity(*m_pcPosLight,
               POS_2,
               CQuaternion());
   }
}

/****************************************/
/****************************************/

REGISTER_LOOP_FUNCTIONS(CFlockingLoopFunctions, "flocking_loop_functions")
