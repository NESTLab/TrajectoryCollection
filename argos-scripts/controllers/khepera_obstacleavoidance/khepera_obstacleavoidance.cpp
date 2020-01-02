/* Include the controller definition */
#include "khepera_obstacleavoidance.h"
/* Function definitions for XML parsing */
#include <argos3/core/utility/configuration/argos_configuration.h>

/****************************************/
/****************************************/

CKheperaObstacleAvoidance::CKheperaObstacleAvoidance() :
   m_pcWheels(NULL),
   m_pcProximity(NULL),
   m_fWheelVelocity(2.5f),
   m_unRId(0) {}

/****************************************/
/****************************************/

void CKheperaObstacleAvoidance::Init(TConfigurationNode& t_node) {
   /*
    * Get sensor/actuator handles
    *
    * The passed string (ex. "differential_steering") corresponds to the
    * XML tag of the device whose handle we want to have. For a list of
    * allowed values, type at the command prompt:
    *
    * $ argos3 -q actuators
    *
    * to have a list of all the possible actuators, or
    *
    * $ argos3 -q sensors
    *
    * to have a list of all the possible sensors.
    *
    * NOTE: ARGoS creates and initializes actuators and sensors
    * internally, on the basis of the lists provided the configuration
    * file at the <controllers><khepera_obstacleavoidance><actuators> and
    * <controllers><khepera_obstacleavoidance><sensors> sections. If you forgot to
    * list a device in the XML and then you request it here, an error
    * occurs.
    */
   m_pcRABAct    = GetActuator<CCI_RangeAndBearingActuator  >("range_and_bearing" );
   m_pcRABSens   = GetSensor  <CCI_RangeAndBearingSensor    >("range_and_bearing" );
   m_pcWheels    = GetActuator<CCI_DifferentialSteeringActuator>("differential_steering");
   m_pcProximity = GetSensor  <CCI_KheperaIVProximitySensor             >("kheperaiv_proximity"    );

   /*
    * Parse the configuration file
    *
    * The user defines this part. Here, the algorithm accepts three
    * parameters and it's nice to put them in the config file so we don't
    * have to recompile if we want to try other settings.
    */
   GetNodeAttributeOrDefault(t_node, "velocity", m_fWheelVelocity, m_fWheelVelocity);
   /* Update Robot Id*/
   const std::string& strRobotId = GetId();
   m_unRId = static_cast<UInt8>(FromString<int>(strRobotId.substr(2)));
}

/****************************************/
/****************************************/

void CKheperaObstacleAvoidance::ControlStep() {
   /* Get the highest reading in front of the robot, which corresponds to the closest object */
   const CCI_KheperaIVProximitySensor::TReadings& tProxReads = m_pcProximity->GetReadings();//[0];
   UInt32 unMaxReadIdx = 0;
   Real fMaxReadVal = tProxReads[unMaxReadIdx].Value;
   if(fMaxReadVal < tProxReads[1].Value) {
      fMaxReadVal = tProxReads[1].Value;
      unMaxReadIdx = 1;
   }
   if(fMaxReadVal < tProxReads[7].Value) {
      fMaxReadVal = tProxReads[7].Value;
      unMaxReadIdx = 7;
   }
   if(fMaxReadVal < tProxReads[6].Value) {
      fMaxReadVal = tProxReads[6].Value;
      unMaxReadIdx = 6;
   }
   /* Do we have an obstacle in front? */
   if(fMaxReadVal > 0.0f) {
     /* Yes, we do: avoid it */
     if(unMaxReadIdx == 0 || unMaxReadIdx == 1) {
       /* The obstacle is on the left, turn right */
       m_pcWheels->SetLinearVelocity(m_fWheelVelocity, 0.0f);
     }
     else {
       /* The obstacle is on the left, turn right */
       m_pcWheels->SetLinearVelocity(0.0f, m_fWheelVelocity);
     }
   }
   else {
     /* No, we don't: go straight */
      m_pcWheels->SetLinearVelocity(m_fWheelVelocity, m_fWheelVelocity);
   }
   m_pcRABAct->SetData(0, m_unRId);
}

/****************************************/
/****************************************/

/*
 * This statement notifies ARGoS of the existence of the controller.
 * It binds the class passed as first argument to the string passed as
 * second argument.
 * The string is then usable in the configuration file to refer to this
 * controller.
 * When ARGoS reads that string in the configuration file, it knows which
 * controller class to instantiate.
 * See also the configuration files for an example of how this is used.
 */
REGISTER_CONTROLLER(CKheperaObstacleAvoidance, "khepera_obstacleavoidance_controller")
