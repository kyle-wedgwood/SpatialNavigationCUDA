# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.4

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation"

# Include any dependencies generated for this target.
include src/CMakeFiles/FOO_CORE.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/FOO_CORE.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/FOO_CORE.dir/flags.make

src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.o: src/CMakeFiles/FOO_CORE.dir/flags.make
src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.o: src/testVisualisation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.o"
	cd "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.o -c "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation/src/testVisualisation.cpp"

src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.i"
	cd "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation/src/testVisualisation.cpp" > CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.i

src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.s"
	cd "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation/src/testVisualisation.cpp" -o CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.s

src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.o.requires:

.PHONY : src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.o.requires

src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.o.provides: src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/FOO_CORE.dir/build.make src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.o.provides.build
.PHONY : src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.o.provides

src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.o.provides.build: src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.o


FOO_CORE: src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.o
FOO_CORE: src/CMakeFiles/FOO_CORE.dir/build.make

.PHONY : FOO_CORE

# Rule to build all files generated by this target.
src/CMakeFiles/FOO_CORE.dir/build: FOO_CORE

.PHONY : src/CMakeFiles/FOO_CORE.dir/build

src/CMakeFiles/FOO_CORE.dir/requires: src/CMakeFiles/FOO_CORE.dir/testVisualisation.cpp.o.requires

.PHONY : src/CMakeFiles/FOO_CORE.dir/requires

src/CMakeFiles/FOO_CORE.dir/clean:
	cd "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation/src" && $(CMAKE_COMMAND) -P CMakeFiles/FOO_CORE.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/FOO_CORE.dir/clean

src/CMakeFiles/FOO_CORE.dir/depend:
	cd "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation" "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation/src" "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation" "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation/src" "/Users/kcaw201/Dropbox/Spatial Navigation CUDA/Visualisation/src/CMakeFiles/FOO_CORE.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : src/CMakeFiles/FOO_CORE.dir/depend

