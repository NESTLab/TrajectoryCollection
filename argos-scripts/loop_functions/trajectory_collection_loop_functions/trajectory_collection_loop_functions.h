#ifndef TRAJECTORY_COLLECTION_LOOP_FUNCTIONS_H
#define TRAJECTORY_COLLECTION_LOOP_FUNCTIONS_H

#include <argos3/core/simulator/loop_functions.h>
#include <argos3/plugins/robots/kheperaiv/simulator/kheperaiv_entity.h>
#include <argos3/plugins/robots/generic/control_interface/ci_range_and_bearing_sensor.h>
#include <argos3/plugins/simulator/entities/rab_equipped_entity.h>
#include <argos3/core/utility/logging/argos_log.h>
#include <map>
#include <algorithm>

using namespace argos;

class CTrajectoryCollectionLoopFunctions : public CLoopFunctions {

public:

   struct STrajectoryData {
      UInt32 TrackedRobot;
      UInt32 StartTime;
      UInt32 PrevTime;
      CVector3 StartPosition;
      CQuaternion StartOrientation;
      std::vector<CVector3> Waypoints;
      STrajectoryData():
         StartTime(0),
         PrevTime(0){}
   };

   typedef std::map<CKheperaIVEntity*, std::vector<STrajectoryData> > TMapKheperaToTrajectories;


public:

   virtual ~CTrajectoryCollectionLoopFunctions() {}

   virtual void Init(TConfigurationNode& t_tree);

   virtual void Reset();

   virtual void PostStep();

   virtual void PostExperiment();

   inline const TMapKheperaToTrajectories& GetPotentialTrajectories() const {
      return m_tPotentialTrajectories;
   }
   inline const TMapKheperaToTrajectories& GetSavedTrajectories() const {
      return m_tSavedTrajectories;
   }

private:

   /* Output file */
   std::string m_strFilename;
   std::ofstream m_cOutput;
   std::ofstream m_cGraphOutput;
   /* Vector of khepera pointers */
   std::vector<CKheperaIVEntity*> m_pcKheperas;
   /* Vector of RAB sensor pointers */
   std::vector<CCI_RangeAndBearingSensor*> m_pcRABSensors;
   /* Maps of potential and saved trajectories */
   TMapKheperaToTrajectories m_tPotentialTrajectories;
   TMapKheperaToTrajectories m_tSavedTrajectories;

   UInt32 m_unClock;

};

#endif
