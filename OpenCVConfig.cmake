# ===================================================================================
#  The OpenCV CMake configuration file
#
#             ** File generated automatically, do not modify **
#
#  Usage from an external project:
#    In your CMakeLists.txt, add these lines:
#
#    FIND_PACKAGE(OpenCV REQUIRED )
#    TARGET_LINK_LIBRARIES(MY_TARGET_NAME ${OpenCV_LIBS})
#
#    This file will define the following variables:
#      - OpenCV_LIBS                 : The list of libraries to links against.
#      - OpenCV_LIB_DIR              : The directory where lib files are. Calling LINK_DIRECTORIES
#                                      with this path is NOT needed.
#      - OpenCV_INCLUDE_DIRS         : The OpenCV include directories.
#      - OpenCV_COMPUTE_CAPABILITIES : The version of compute capability
#      - OpenCV_VERSION              : The version of this OpenCV build. Example: "2.3.2"
#      - OpenCV_VERSION_MAJOR        : Major version part of OpenCV_VERSION. Example: "2"
#      - OpenCV_VERSION_MINOR        : Minor version part of OpenCV_VERSION. Example: "3"
#      - OpenCV_VERSION_PATCH        : Patch version part of OpenCV_VERSION. Example: "2"
#
#    Advanced variables:
#      - OpenCV_SHARED
#      - OpenCV_CONFIG_PATH
#      - OpenCV_INSTALL_PATH
#      - OpenCV_LIB_COMPONENTS
#      - OpenCV_EXTRA_COMPONENTS
#      - OpenCV_USE_MANGLED_PATHS
#      - OpenCV_HAVE_ANDROID_CAMERA
#      - OpenCV_SOURCE_PATH
#
# =================================================================================================

# ======================================================
# Version Compute Capability from which library OpenCV
# has been compiled is remembered
# ======================================================
SET(OpenCV_COMPUTE_CAPABILITIES -gencode;arch=compute_11,code=sm_11;-gencode;arch=compute_12,code=sm_12;-gencode;arch=compute_13,code=sm_13;-gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_20,code=sm_21;-gencode;arch=compute_20,code=compute_20)

# Some additional settings are required if OpenCV is built as static libs
set(OpenCV_SHARED ON)

# Enables mangled install paths, that help with side by side installs
set(OpenCV_USE_MANGLED_PATHS )

# Extract the directory where *this* file has been installed (determined at cmake run-time)
get_filename_component(OpenCV_CONFIG_PATH "${CMAKE_CURRENT_LIST_FILE}" PATH)

# Get the absolute path with no ../.. relative marks, to eliminate implicit linker warnings
get_filename_component(OpenCV_INSTALL_PATH "${OpenCV_CONFIG_PATH}/../.." REALPATH)

# Presence of Android native camera support
set (OpenCV_HAVE_ANDROID_CAMERA )

# ======================================================
# Include directories to add to the user project:
# ======================================================

# Provide the include directories to the caller
SET(OpenCV_INCLUDE_DIRS "C:/Users/Andrey/Documents/Visual Studio 2010/Projects/OpenCV_sln" "C:/Users/Andrey/Documents/opencv/include" "C:/Users/Andrey/Documents/opencv/include/opencv")
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# ======================================================
# Link directories to add to the user project:
# ======================================================

# Provide the libs directory anyway, it may be needed in some cases.
SET(OpenCV_LIB_DIR "C:/Users/Andrey/Documents/Visual Studio 2010/Projects/OpenCV_sln/lib")
LINK_DIRECTORIES(${OpenCV_LIB_DIR})

# ====================================================================
# Link libraries: e.g.   libopencv_core.so, opencv_imgproc220d.lib, etc...
# ====================================================================

# OpenCV internal dependencies:
# opencv_androidcamera -> {}
# opencv_core          -> {}
# opencv_flann         -> {opencv_core}
# opencv_imgproc       -> {opencv_core}
# opencv_ml            -> {opencv_core}
# opencv_highgui       -> {opencv_core, opencv_imgproc, opencv_androidcamera}
# opencv_video         -> {opencv_core, opencv_imgproc}
# opencv_features2d    -> {opencv_core, opencv_imgproc, opencv_flann, opencv_highgui}
# opencv_calib3d       -> {opencv_core, opencv_imgproc, opencv_flann,                 opencv_features2d}
# opencv_objdetect     -> {opencv_core, opencv_imgproc, opencv_flann, opencv_highgui, opencv_features2d, opencv_calib3d}
# opencv_gpu           -> {opencv_core, opencv_imgproc, opencv_flann,                 opencv_features2d, opencv_calib3d, opencv_objdetect}
# opencv_stitching     -> {opencv_core, opencv_imgproc, opencv_flann,                 opencv_features2d, opencv_calib3d, opencv_objdetect, opencv_gpu}
# opencv_legacy        -> {opencv_core, opencv_imgproc, opencv_flann, opencv_highgui, opencv_features2d, opencv_calib3d,                   opencv_video}
# opencv_contrib       -> {opencv_core, opencv_imgproc, opencv_flann, opencv_highgui, opencv_features2d, opencv_calib3d, opencv_objdetect, opencv_video, opencv_ml}

SET(OpenCV_LIB_COMPONENTS opencv_contrib opencv_legacy opencv_stitching opencv_gpu opencv_objdetect opencv_calib3d opencv_features2d opencv_video opencv_highgui opencv_ml opencv_imgproc opencv_flann opencv_core opencv_androidcamera)

# remove modules unavailable on current platform:
if(ANDROID)
    LIST(REMOVE_ITEM OpenCV_LIB_COMPONENTS opencv_gpu)
    SET(OpenCV_LIB_ANDROID )
    IF(OpenCV_LIB_ANDROID)
        SET(OpenCV_LIB_COMPONENTS ${OpenCV_LIB_COMPONENTS} ${OpenCV_LIB_ANDROID})
        SET(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
        SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
    ENDIF()
endif()
if(NOT ANDROID OR OpenCV_SHARED OR NOT OpenCV_HAVE_ANDROID_CAMERA)
    LIST(REMOVE_ITEM OpenCV_LIB_COMPONENTS opencv_androidcamera)
endif()

if(OpenCV_USE_MANGLED_PATHS)
      #be explicit about the library names.
      set(OpenCV_LIB_COMPONENTS_ )
      foreach( CVLib ${OpenCV_LIB_COMPONENTS})
        list(APPEND OpenCV_LIB_COMPONENTS_ ${OpenCV_LIB_DIR}/lib${CVLib}.so.2.3.2 )
      endforeach()
      set(OpenCV_LIB_COMPONENTS ${OpenCV_LIB_COMPONENTS_})
endif()

SET(OpenCV_LIBS "")
if(WIN32)
  foreach(__CVLIB ${OpenCV_LIB_COMPONENTS})
      # CMake>=2.6 supports the notation "debug XXd optimized XX"
      if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} VERSION_GREATER 2.4)
          # Modern CMake:
          SET(OpenCV_LIBS ${OpenCV_LIBS} debug ${__CVLIB}232d optimized ${__CVLIB}232)
      else()
          # Old CMake:
          SET(OpenCV_LIBS ${OpenCV_LIBS} ${__CVLIB}232)
      endif()
  endforeach()
else()
  foreach(__CVLIB ${OpenCV_LIB_COMPONENTS})
    SET(OpenCV_LIBS ${OpenCV_LIBS} ${__CVLIB})
  endforeach()
endif()

# ==============================================================
#  Extra include directories, needed by OpenCV 2 new structure
# ==============================================================
SET(OpenCV_SOURCE_PATH "C:/Users/Andrey/Documents/opencv")
if(NOT "${OpenCV_SOURCE_PATH}" STREQUAL  "")
    foreach(__CVLIB ${OpenCV_LIB_COMPONENTS})
        # We only need the "core",... part here: "opencv_core" -> "core"
        STRING(REGEX REPLACE "opencv_(.*)" "\\1" __MODNAME ${__CVLIB})
        INCLUDE_DIRECTORIES("${OpenCV_SOURCE_PATH}/modules/${__MODNAME}/include")
        LIST(APPEND OpenCV_INCLUDE_DIRS "${OpenCV_SOURCE_PATH}/modules/${__MODNAME}/include")
    endforeach()
endif()

# For OpenCV built as static libs, we need the user to link against
#  many more dependencies:
IF (NOT OpenCV_SHARED)
    # Under static libs, the user of OpenCV needs access to the 3rdparty libs as well:
    LINK_DIRECTORIES("C:/Users/Andrey/Documents/Visual Studio 2010/Projects/OpenCV_sln/3rdparty/lib")

    set(OpenCV_LIBS glu32;opengl32  comctl32;gdi32;ole32;vfw32 ${OpenCV_LIBS})
    set(OpenCV_EXTRA_COMPONENTS libjpeg libpng libtiff libjasper zlib)

    if (WIN32 AND ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} VERSION_GREATER 2.4)
        # Modern CMake:
        foreach(__EXTRA_LIB ${OpenCV_EXTRA_COMPONENTS})
            set(OpenCV_LIBS ${OpenCV_LIBS}
                debug ${__EXTRA_LIB}d
                optimized ${__EXTRA_LIB})
        endforeach()
    else()
        # Old CMake:
        set(OpenCV_LIBS ${OpenCV_LIBS} ${OpenCV_EXTRA_COMPONENTS})
    endif()

    if (APPLE)
        set(OpenCV_LIBS ${OpenCV_LIBS} "-lbz2" "-framework Cocoa" "-framework QuartzCore" "-framework QTKit")
    endif()
ENDIF()

# ======================================================
#  Android camera helper macro
# ======================================================
IF (OpenCV_HAVE_ANDROID_CAMERA)
  macro( COPY_NATIVE_CAMERA_LIBS target )
    get_target_property(target_location ${target} LOCATION)
    get_filename_component(target_location "${target_location}" PATH)
    file(GLOB camera_wrappers "${OpenCV_LIB_DIR}/libnative_camera_r*.so")
    foreach(wrapper ${camera_wrappers})
      ADD_CUSTOM_COMMAND(
        TARGET ${target}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${wrapper}" "${target_location}"
      )
    endforeach()
  endmacro()
ENDIF()

# ======================================================
#  Version variables:
# ======================================================
SET(OpenCV_VERSION 2.3.2)
SET(OpenCV_VERSION_MAJOR  2)
SET(OpenCV_VERSION_MINOR  3)
SET(OpenCV_VERSION_PATCH  2)
