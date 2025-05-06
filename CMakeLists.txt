# Create a custom library
add_library(
  monitors_module
  monitors_module/AppKey.cc
  monitors_module/PacketKey.cc
  monitors_module/Monitor.cc
  monitors_module/E2EMonitor.cc
  monitors_module/SwitchMonitor.cc
  monitors_module/PoissonSampler.cc
  monitors_module/RegularSampler.cc
  monitors_module/NetDeviceMonitor.cc
  monitors_module/BurstMonitor.cc
)

add_library(
  helper_classes
  helper_classes/HelperMethods.cc
)

add_library(
  traffic_generator_module
  traffic_generator_module/background_replay/BackgroundReplay.cc
  traffic_generator_module/background_replay/TraceReplayReceiver.cc
  traffic_generator_module/background_replay/TraceReplayReceiverHelper.cc
  traffic_generator_module/background_replay/TraceReplaySender.cc
  traffic_generator_module/background_replay/TraceReplaySenderHelper.cc
  traffic_generator_module/DC_traffic_generator/DCWorkloadGenerator.cc
  traffic_generator_module/DC_traffic_generator/WorkloadApp.cc
  traffic_generator_module/DC_traffic_generator/ConnectionPool.cc
)

# Link the custom library to the scratch executable
build_exec(
  EXECNAME DatacenterSimulation
  SOURCE_FILES DatacenterSimulation.cc
  LIBRARIES_TO_LINK monitors_module
                    traffic_generator_module
                    helper_classes
                    ${libcore}
                    ${ns3-libs}
  # use "${ns3-libs}" "${ns3-contrib-libs}" in case you want to link to all
  # enabled modules
  EXECUTABLE_DIRECTORY_PATH ${CMAKE_OUTPUT_DIRECTORY}/scratch/ECNMC/
)
