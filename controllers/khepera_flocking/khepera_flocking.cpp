/* Include the controller definition */
#include "khepera_flocking.h"
/* Function definitions for XML parsing */
#include <argos3/core/utility/configuration/argos_configuration.h>
#include <argos3/core/utility/logging/argos_log.h>

/****************************************/
/****************************************/

void CKheperaFlocking::SWheelTurningParams::Init(TConfigurationNode& t_node) {
   try {
      TurningMechanism = NO_TURN;
      CDegrees cAngle;
      GetNodeAttribute(t_node, "hard_turn_angle_threshold", cAngle);
      HardTurnOnAngleThreshold = ToRadians(cAngle);
      GetNodeAttribute(t_node, "soft_turn_angle_threshold", cAngle);
      SoftTurnOnAngleThreshold = ToRadians(cAngle);
      GetNodeAttribute(t_node, "no_turn_angle_threshold", cAngle);
      NoTurnAngleThreshold = ToRadians(cAngle);
      GetNodeAttribute(t_node, "max_speed", MaxSpeed);
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error initializing controller wheel turning parameters.", ex);
   }
}

/****************************************/
/****************************************/

void CKheperaFlocking::SFlockingInteractionParams::Init(TConfigurationNode& t_node) {
   try {
      GetNodeAttribute(t_node, "target_distance", TargetDistance);
      GetNodeAttribute(t_node, "gain", Gain);
      GetNodeAttribute(t_node, "exponent", Exponent);
      GetNodeAttribute(t_node, "max_interaction", MaxInteraction);
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error initializing controller flocking parameters.", ex);
   }
}

/****************************************/
/****************************************/

/*
 * This function is a generalization of the Lennard-Jones potential
 */
Real CKheperaFlocking::SFlockingInteractionParams::GeneralizedLennardJones(Real f_distance) {
   Real fNormDistExp = ::pow(TargetDistance / f_distance, Exponent);
   return -Gain / f_distance * (fNormDistExp * fNormDistExp - fNormDistExp);
}

/****************************************/
/****************************************/

CKheperaFlocking::CKheperaFlocking() :
   m_pcWheels(NULL),
   m_pcLight(NULL),
   m_pcRABAct(NULL),
   m_pcRABSens(NULL),
   m_unRId(0) {}

/****************************************/
/****************************************/

void CKheperaFlocking::Init(TConfigurationNode& t_node) {
   /*
    * Get sensor/actuator handles
    *
    * The passed string (ex. "differential_steering") corresponds to the XML tag of the
    * device whose handle we want to have. For a list of allowed values, type at the
    * command prompt:
    *
    * $ argos3 -q actuators
    *
    * to have a list of all the possible actuators, or
    *
    * $ argos3 -q sensors
    *
    * to have a list of all the possible sensors.
    *
    * NOTE: ARGoS creates and initializes actuators and sensors internally, on the basis of
    *       the lists provided the configuration file at the
    *       <controllers><khepera_diffusion><actuators> and
    *       <controllers><khepera_diffusion><sensors> sections. If you forgot to
    *       list a device in the XML and then you request it here, an error occurs.
    */
   m_pcRABAct    = GetActuator<CCI_RangeAndBearingActuator  >("range_and_bearing" );
   m_pcRABSens   = GetSensor  <CCI_RangeAndBearingSensor    >("range_and_bearing" );
   m_pcWheels = GetActuator<CCI_DifferentialSteeringActuator          >("differential_steering");
   m_pcLight  = GetSensor  <CCI_KheperaIVLightSensor                    >("kheperaiv_light");
   /*
    * Parse the config file
    */
   try {
      /* Wheel turning */
      m_sWheelTurningParams.Init(GetNode(t_node, "wheel_turning"));
      /* Flocking-related */
      m_sFlockingParams.Init(GetNode(t_node, "flocking"));
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error parsing the controller parameters.", ex);
   }
   /* Update Robot Id*/
   const std::string& strRobotId = GetId();
   m_unRId = FromString<UInt8>(strRobotId.substr(1));
}

/****************************************/
/****************************************/

void CKheperaFlocking::ControlStep() {
   SetWheelSpeedsFromVector(VectorToLight() + FlockingVector());
   m_pcRABAct->SetData(0, m_unRId);
}

/****************************************/
/****************************************/

CVector2 CKheperaFlocking::VectorToLight() {
   /* Get light readings */
   const CCI_KheperaIVLightSensor::TReadings& tReadings = m_pcLight->GetReadings();
   /* Calculate a normalized vector that points to the closest light */
   CVector2 cAccum;
   for(size_t i = 0; i < tReadings.size(); ++i) {
      cAccum += CVector2(tReadings[i].Value, tReadings[i].Angle);
   }
   if(cAccum.Length() > 0.0f) {
      /* Make the vector long as 1/4 of the max speed */
      cAccum.Normalize();
      cAccum *= 0.25f * m_sWheelTurningParams.MaxSpeed;
   }
   return cAccum;
}

/****************************************/
/****************************************/

CVector2 CKheperaFlocking::FlockingVector() {
   /* Get RAB messages from nearby kheperas */
   const CCI_RangeAndBearingSensor::TReadings& tMsgs = m_pcRABSens->GetReadings();
   /* Go through them to calculate the flocking interaction vector */
   if(! tMsgs.empty()) {
      /* This will contain the final interaction vector */
      CVector2 cAccum;
      /* Used to calculate the vector length of each neighbor's contribution */
      Real fLJ;
      /* A counter for the neighbors in state flock */
      UInt32 unPeers = 0;
      for(size_t i = 0; i < tMsgs.size(); ++i) {
         if(tMsgs[i].Data[0] > 0) {
            /*
             * Take the message sender range and horizontal bearing
             * With the range, calculate the Lennard-Jones interaction force
             * Form a 2D vector with the interaction force and the bearing
             * Sum such vector to the accumulator
             */
            /* Calculate LJ */
            fLJ = m_sFlockingParams.GeneralizedLennardJones(tMsgs[i].Range);
            /* Sum to accumulator */
            cAccum += CVector2(fLJ,
                               tMsgs[i].HorizontalBearing);
            /* Count one more flocking neighbor */
            ++unPeers;
         }
      }
      if(unPeers > 0) {
         /* Divide the accumulator by the number of flocking neighbors */
         cAccum /= unPeers;
         /* Limit the interaction force */
         if(cAccum.Length() > m_sFlockingParams.MaxInteraction) {
            cAccum.Normalize();
            cAccum *= m_sFlockingParams.MaxInteraction;
         }
      }
      /* All done */
      return cAccum;
   }
   else {
      /* No messages received, no interaction */
      return CVector2();
   }
}

/****************************************/
/****************************************/

void CKheperaFlocking::SetWheelSpeedsFromVector(const CVector2& c_heading) {
   /* Get the heading angle */
   CRadians cHeadingAngle = c_heading.Angle().SignedNormalize();
   /* Get the length of the heading vector */
   Real fHeadingLength = c_heading.Length();
   /* Clamp the speed so that it's not greater than MaxSpeed */
   Real fBaseAngularWheelSpeed = Min<Real>(fHeadingLength, m_sWheelTurningParams.MaxSpeed);
   /* State transition logic */
   if(m_sWheelTurningParams.TurningMechanism == SWheelTurningParams::HARD_TURN) {
      if(Abs(cHeadingAngle) <= m_sWheelTurningParams.SoftTurnOnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::SOFT_TURN;
      }
   }
   if(m_sWheelTurningParams.TurningMechanism == SWheelTurningParams::SOFT_TURN) {
      if(Abs(cHeadingAngle) > m_sWheelTurningParams.HardTurnOnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::HARD_TURN;
      }
      else if(Abs(cHeadingAngle) <= m_sWheelTurningParams.NoTurnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::NO_TURN;
      }
   }
   if(m_sWheelTurningParams.TurningMechanism == SWheelTurningParams::NO_TURN) {
      if(Abs(cHeadingAngle) > m_sWheelTurningParams.HardTurnOnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::HARD_TURN;
      }
      else if(Abs(cHeadingAngle) > m_sWheelTurningParams.NoTurnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::SOFT_TURN;
      }
   }
   /* Wheel speeds based on current turning state */
   Real fSpeed1, fSpeed2;
   switch(m_sWheelTurningParams.TurningMechanism) {
      case SWheelTurningParams::NO_TURN: {
         /* Just go straight */
         fSpeed1 = fBaseAngularWheelSpeed;
         fSpeed2 = fBaseAngularWheelSpeed;
         break;
      }
      case SWheelTurningParams::SOFT_TURN: {
         /* Both wheels go straight, but one is faster than the other */
         Real fSpeedFactor = (m_sWheelTurningParams.HardTurnOnAngleThreshold - Abs(cHeadingAngle)) / m_sWheelTurningParams.HardTurnOnAngleThreshold;
         fSpeed1 = fBaseAngularWheelSpeed - fBaseAngularWheelSpeed * (1.0 - fSpeedFactor);
         fSpeed2 = fBaseAngularWheelSpeed + fBaseAngularWheelSpeed * (1.0 - fSpeedFactor);
         break;
      }
      case SWheelTurningParams::HARD_TURN: {
         /* Opposite wheel speeds */
         fSpeed1 = -m_sWheelTurningParams.MaxSpeed;
         fSpeed2 =  m_sWheelTurningParams.MaxSpeed;
         break;
      }
   }
   /* Apply the calculated speeds to the appropriate wheels */
   Real fLeftWheelSpeed, fRightWheelSpeed;
   if(cHeadingAngle > CRadians::ZERO) {
      /* Turn Left */
      fLeftWheelSpeed  = fSpeed1;
      fRightWheelSpeed = fSpeed2;
   }
   else {
      /* Turn Right */
      fLeftWheelSpeed  = fSpeed2;
      fRightWheelSpeed = fSpeed1;
   }
   /* Finally, set the wheel speeds */
   m_pcWheels->SetLinearVelocity(fLeftWheelSpeed, fRightWheelSpeed);
}

/****************************************/
/****************************************/

/*
 * This statement notifies ARGoS of the existence of the controller.
 * It binds the class passed as first argument to the string passed as second argument.
 * The string is then usable in the XML configuration file to refer to this controller.
 * When ARGoS reads that string in the XML file, it knows which controller class to instantiate.
 * See also the XML configuration files for an example of how this is used.
 */
REGISTER_CONTROLLER(CKheperaFlocking, "khepera_flocking_controller")
