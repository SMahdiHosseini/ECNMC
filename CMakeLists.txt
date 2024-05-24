##################### Part here is to help compile within the ns3 project  #####################
# set(target_name Edge-Core-Network-Measurement-Consistency)
# set(output_directory ${CMAKE_OUTPUT1DIRECTORY}/scratch/Edge-Core-Network-Measurement-Consistency/)
# set(target_prefix scratch_Edge-Core-Network-Measurement-Consistency_)

# project(${target_prefix}${target_name})
# ################################################################################################


# add_definitions(-DNS3_LOG_ENABLE)
# set(NS3_BUILD_PATH  /home/mahdi/Documents/NAL/ns-3-dev/build)
# include_directories(${NS3_BUILD_PATH}/include)
# link_directories(${NS3_BUILD_PATH}/include)
# include_directories(${NS3_BUILD_PATH}/build)
# link_directories(${NS3_BUILD_PATH}/build)

# file(GLOB SOURCE_FILES N4_datacenter_switch.cc)
# file(GLOB MONITOR_FILES monitors_module/*)


# add_executable(${PROJECT_NAME}
#         ${SOURCE_FILES}
#         ${MONITOR_FILES})


# set(NS3_EMU ON)
# target_link_libraries(${PROJECT_NAME}
#         libcore
#         libnetanim
#         libwifi
#         libnetwork
#         libinternet
#         libapplications
#         libpoint-to-point)

# ##################### Part here is to help compile within the ns3 project  #####################
# set(ns3-exec-outputname ns${NS3_VER}-${target_name}${build_profile_suffix})
# set(ns3-execs "${output_directory}${ns3-exec-outputname};${ns3-execs}" CACHE INTERNAL "list of c++ executables")

# set_target_properties(
#         ${PROJECT_NAME}
#         PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${output_directory}
#                    RUNTIME_OUTPUT_NAME ${ns3-exec-outputname}
# )
################################################################################################

# Create a custom library
add_library(
  monitors_module
  monitors_module/AppKey.cc
  monitors_module/PacketKey.cc
  monitors_module/PacketMonitor.cc
  monitors_module/SwitchMonitor.cc
  monitors_module/PoissonSampler.cc
  monitors_module/RegularSampler.cc
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
)

# Link the custom library to the scratch executable
build_exec(
  EXECNAME N4_datacenter_switch
  SOURCE_FILES N4_datacenter_switch.cc
  LIBRARIES_TO_LINK monitors_module
                    traffic_generator_module
                    helper_classes
                    ${libcore}
                    ${ns3-libs}
  # use "${ns3-libs}" "${ns3-contrib-libs}" in case you want to link to all
  # enabled modules
  EXECUTABLE_DIRECTORY_PATH ${CMAKE_OUTPUT_DIRECTORY}/scratch/ECNMC/
)
