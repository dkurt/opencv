# The script detects Intel(R) Inference Engine installation
#
# Cache variables:
# INF_ENGINE_RELEASE - a number reflecting IE source interface (linked with OpenVINO release)
#
# Detect parameters:
# 1. Native cmake IE package:
#    - environment variable InferenceEngine_DIR is set to location of cmake module
# 2. Custom location:
#    - INF_ENGINE_INCLUDE_DIRS - headers search location
#    - INF_ENGINE_LIB_DIRS     - library search location
# 3. OpenVINO location:
#    - environment variable INTEL_OPENVINO_DIR is set to location of OpenVINO installation dir
#    - INF_ENGINE_PLATFORM - part of name of library directory representing its platform
#
# Result:
# INF_ENGINE_TARGET - set to name of imported library target representing InferenceEngine
#

if(NOT HAVE_CXX11)
    message(WARNING "DL Inference engine requires C++11. You can turn it on via ENABLE_CXX11=ON CMake flag.")
    return()
endif()

# =======================

function(add_custom_ie_build _inc _lib _lib_rel _lib_dbg _msg)
  if(NOT _inc OR NOT (_lib OR _lib_rel OR _lib_dbg))
    return()
  endif()
  add_library(inference_engine UNKNOWN IMPORTED)
  set_target_properties(inference_engine PROPERTIES
    IMPORTED_LOCATION "${_lib}"
    IMPORTED_IMPLIB_RELEASE "${_lib_rel}"
    IMPORTED_IMPLIB_DEBUG "${_lib_dbg}"
    INTERFACE_INCLUDE_DIRECTORIES "${_inc}"
  )
  if(NOT INF_ENGINE_RELEASE VERSION_GREATER "2018050000")
    find_library(INF_ENGINE_OMP_LIBRARY iomp5 PATHS "${INF_ENGINE_OMP_DIR}" NO_DEFAULT_PATH)
    if(NOT INF_ENGINE_OMP_LIBRARY)
      message(WARNING "OpenMP for IE have not been found. Set INF_ENGINE_OMP_DIR variable if you experience build errors.")
    else()
      set_target_properties(inference_engine PROPERTIES IMPORTED_LINK_INTERFACE_LIBRARIES "${INF_ENGINE_OMP_LIBRARY}")
    endif()
  endif()
  set(INF_ENGINE_VERSION "Unknown" CACHE STRING "")
  set(INF_ENGINE_TARGET inference_engine PARENT_SCOPE)
  message(STATUS "Detected InferenceEngine: ${_msg}")
endfunction()

# ======================

function(download_ie)
  set(ie_src_dir "${OpenCV_BINARY_DIR}/3rdparty/dldt")
  set(ie_subdir "dldt-2019_R2")
  ocv_download(FILENAME "2019_R2.zip"
               HASH "c486380deb75503adca5d7e88763e4a5"
               URL
                 "${OPENCV_IE_URL}"
                 "$ENV{OPENCV_IE_URL}"
                 "https://github.com/opencv/dldt/archive/"
               DESTINATION_DIR ${ie_src_dir}
               ID IE
               STATUS res
               UNPACK RELATIVE_URL)

  if (NOT res)
      return()
  endif()

  function(ie_patch filename from to)
    file(READ "${ie_src_dir}/${ie_subdir}/inference-engine/${filename}" filedata)
    string(REPLACE "${from}" "${to}" filedata "${filedata}")
    file(WRITE "${ie_src_dir}/${ie_subdir}/inference-engine/${filename}" "${filedata}")
  endfunction()

  # This is a minor patch to IE's cmake files.
  ie_patch("cmake/dependencies.cmake" "CMAKE_SOURCE_DIR" "PROJECT_SOURCE_DIR")
  ie_patch("src/inference_engine/CMakeLists.txt" "CMAKE_SOURCE_DIR" "PROJECT_SOURCE_DIR")
  ie_patch("CMakeLists.txt" "CMAKE_SOURCE_DIR" "PROJECT_SOURCE_DIR")

  ie_patch("CMakeLists.txt" "add_subdirectory(tests)" "")  # Disable tests
  ie_patch("CMakeLists.txt" "add_subdirectory(samples)" "")  # Disable samples
  ie_patch("CMakeLists.txt" "add_subdirectory(tools)" "")  # Disable tools
  # Enable extensions library
  ie_patch("src/CMakeLists.txt" "add_subdirectory(extension EXCLUDE_FROM_ALL)" "add_subdirectory(extension)")
  # Redirect plugins.xml output
  ie_patch("cmake/plugins/plugins.cmake" "IE_CONFIG_OUTPUT_FILE=\${config_output_file}" "IE_CONFIG_OUTPUT_FILE=${LIBRARY_OUTPUT_PATH}/plugins.xml")

  function(download_from_github name version md5 org)
    string(TOUPPER ${name} upName)
    set(src_dir "${ie_src_dir}/${ie_subdir}/inference-engine/thirdparty/${name}")
    set(filename "v${version}.zip")
    set(subdir "${name}-${version}")
    ocv_download(FILENAME ${filename}
                 HASH ${md5}
                 URL
                   "${OPENCV_${upName}_URL}"
                   "$ENV{OPENCV_${upName}_URL}"
                   "https://github.com/${org}/${name}/archive/"
                 DESTINATION_DIR ${src_dir}
                 ID ${upName}
                 STATUS res
    UNPACK RELATIVE_URL)

    if (NOT res)
        return()
    endif()

    # Move folder excluding subfolder name
    if(EXISTS "${src_dir}/${subdir}")
      file(RENAME "${src_dir}/${subdir}" "${src_dir}_tmp")
      file(REMOVE_RECURSE "${src_dir}")
      file(RENAME "${src_dir}_tmp" "${src_dir}")
    endif()
  endfunction()

  download_from_github(ade "0.1.1d" "37479d90e3a5d47f132f512b22cbe206" "opencv")
  download_from_github(ngraph "0.22.0" "1ad1e35c1746c67264c9ee525f5893c0" "NervanaSystems")

  set(ENABLE_GNA OFF)
  set(ENABLE_PROFILING_ITT OFF)
  set(ENABLE_SAMPLES_CORE OFF)
  set(ENABLE_SEGMENTATION_TESTS OFF)
  set(ENABLE_OBJECT_DETECTION_TESTS OFF)
  set(ENABLE_OPENCV OFF)
  set(BUILD_TESTS OFF)  # pugixml
  set(BUILD_SHARED_LIBS OFF)  # pugixml

  if(MSVC)
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4146 /wd4703)
  else()
    ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-deprecated -Wmissing-prototypes -Wmissing-declarations -Wshadow
                                         -Wunused-parameter -Wsign-compare -Wstrict-prototypes -Wnon-virtual-dtor
                                         -Wundef -Wstrict-aliasing -Wsign-promo -Wreorder -Wunused-variable
                                         -Wunknown-pragmas -Wstrict-overflow -Wextra -Wunused-local-typedefs
                                         -Wunused-function -Wsequence-point -Wunused-but-set-variable -Wparentheses
                                         -Wsuggest-override -Wimplicit-fallthrough -Wattributes)
    ocv_warnings_disable(CMAKE_C_FLAGS -Wstrict-prototypes)

    if(APPLE)
      ocv_warnings_disable(CMAKE_CXX_FLAGS -Wpessimizing-move -Wunused-private-field
                                           -Wmissing-braces -Wunused-lambda-capture)
    endif()
  endif()

  add_subdirectory(${OpenCV_BINARY_DIR}/3rdparty/dldt/${ie_subdir}/inference-engine
                   ${OpenCV_BINARY_DIR}/3rdparty/dldt)
  if (TARGET MKLDNNPlugin)
    set_target_properties(MKLDNNPlugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}")
    set_target_properties(ie_cpu_extension PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}")
  endif()
  if (TARGET clDNNPlugin)
    set_target_properties(clDNNPlugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}")
  endif()
  if (TARGET myriadPlugin)
    set_target_properties(myriadPlugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}")
  endif()

  set(INF_ENGINE_TARGET inference_engine_s PARENT_SCOPE)
  set(INF_ENGINE_RELEASE "2019020000" PARENT_SCOPE)

  # if(WITH_TBB)
  #   # message("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
  #   # message("${ie_src_dir}/${ie_subdir}/inference-engine/temp/tbb/include")
  #   # find_path(ie_custom_inc "inference_engine.hpp" PATHS "${INF_ENGINE_INCLUDE_DIRS}" NO_DEFAULT_PATH)
  #   unset(TBB_ENV_INCLUDE CACHE)
  #   # message("${TBB_ENV_LIB}")
  #   # message("${TBB_ENV_LIB_DEBUG}")
  #   unset(TBB_ENV_LIB CACHE)
  #   unset(TBB_ENV_LIB_DEBUG CACHE)
  #   unset(HAVE_TBB)
  #
  #   set(ENV{TBBROOT} "${ie_src_dir}/${ie_subdir}/inference-engine/temp/tbb/")
  #   set(TBB_ENV_INCLUDE "${ie_src_dir}/${ie_subdir}/inference-engine/temp/tbb/include")
  #   set(TBB_ENV_LIB "${ie_src_dir}/${ie_subdir}/inference-engine/temp/tbb/lib/libtbb.so")
  #   set(TBB_ENV_LIB_DEBUG "${ie_src_dir}/${ie_subdir}/inference-engine/temp/tbb/lib/libtbb_debug.so")
  #   message("++++++++++++++++++++++++++")
  #   message("${TBB_ENV_INCLUDE}")
  #   include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectTBB.cmake")
  #   # find_path(TBB_ENV_INCLUDE NAMES "tbb/tbb.h" PATHS "${ie_src_dir}/${ie_subdir}/inference-engine/temp/tbb/include" NO_DEFAULT_PATH)
  #   # message("${TBB_ENV_INCLUDE}")
  # endif()
endfunction()

# ======================

find_package(InferenceEngine QUIET)
if(InferenceEngine_FOUND)
  set(INF_ENGINE_TARGET ${InferenceEngine_LIBRARIES})
  set(INF_ENGINE_VERSION "${InferenceEngine_VERSION}" CACHE STRING "")
  message(STATUS "Detected InferenceEngine: cmake package")
endif()

if(NOT INF_ENGINE_TARGET AND INF_ENGINE_LIB_DIRS AND INF_ENGINE_INCLUDE_DIRS)
  find_path(ie_custom_inc "inference_engine.hpp" PATHS "${INF_ENGINE_INCLUDE_DIRS}" NO_DEFAULT_PATH)
  find_library(ie_custom_lib "inference_engine" PATHS "${INF_ENGINE_LIB_DIRS}" NO_DEFAULT_PATH)
  find_library(ie_custom_lib_rel "inference_engine" PATHS "${INF_ENGINE_LIB_DIRS}/Release" NO_DEFAULT_PATH)
  find_library(ie_custom_lib_dbg "inference_engine" PATHS "${INF_ENGINE_LIB_DIRS}/Debug" NO_DEFAULT_PATH)
  add_custom_ie_build("${ie_custom_inc}" "${ie_custom_lib}" "${ie_custom_lib_rel}" "${ie_custom_lib_dbg}" "INF_ENGINE_{INCLUDE,LIB}_DIRS")
endif()

set(_loc "$ENV{INTEL_OPENVINO_DIR}")
if(NOT _loc AND DEFINED ENV{INTEL_CVSDK_DIR})
  set(_loc "$ENV{INTEL_CVSDK_DIR}")  # OpenVINO 2018.x
endif()
if(NOT INF_ENGINE_TARGET AND _loc)
  if(NOT INF_ENGINE_RELEASE VERSION_GREATER "2018050000")
    set(INF_ENGINE_PLATFORM_DEFAULT "ubuntu_16.04")
  else()
    set(INF_ENGINE_PLATFORM_DEFAULT "")
  endif()
  set(INF_ENGINE_PLATFORM "${INF_ENGINE_PLATFORM_DEFAULT}" CACHE STRING "InferenceEngine platform (library dir)")
  find_path(ie_custom_env_inc "inference_engine.hpp" PATHS "${_loc}/deployment_tools/inference_engine/include" NO_DEFAULT_PATH)
  find_library(ie_custom_env_lib "inference_engine" PATHS "${_loc}/deployment_tools/inference_engine/lib/${INF_ENGINE_PLATFORM}/intel64" NO_DEFAULT_PATH)
  find_library(ie_custom_env_lib_rel "inference_engine" PATHS "${_loc}/deployment_tools/inference_engine/lib/intel64/Release" NO_DEFAULT_PATH)
  find_library(ie_custom_env_lib_dbg "inference_engine" PATHS "${_loc}/deployment_tools/inference_engine/lib/intel64/Debug" NO_DEFAULT_PATH)
  add_custom_ie_build("${ie_custom_env_inc}" "${ie_custom_env_lib}" "${ie_custom_env_lib_rel}" "${ie_custom_env_lib_dbg}" "OpenVINO (${_loc})")
endif()

if(NOT INF_ENGINE_TARGET)
  download_ie()
endif()

# Add more features to the target

if(INF_ENGINE_TARGET)
  if(NOT INF_ENGINE_RELEASE)
    message(WARNING "InferenceEngine version have not been set, 2019R2 will be used by default. Set INF_ENGINE_RELEASE variable if you experience build errors.")
  endif()
  set(INF_ENGINE_RELEASE "2019020000" CACHE STRING "Force IE version, should be in form YYYYAABBCC (e.g. 2018R2.0.2 -> 2018020002)")
  set_target_properties(${INF_ENGINE_TARGET} PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "HAVE_INF_ENGINE=1;INF_ENGINE_RELEASE=${INF_ENGINE_RELEASE}"
  )
endif()
