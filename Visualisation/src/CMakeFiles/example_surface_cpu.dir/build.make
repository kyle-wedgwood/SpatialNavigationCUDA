# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/toor/forge/forge-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/toor/forge/forge-master

# Include any dependencies generated for this target.
include examples/CMakeFiles/example_surface_cpu.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/example_surface_cpu.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/example_surface_cpu.dir/flags.make

examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o: examples/CMakeFiles/example_surface_cpu.dir/flags.make
examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o: examples/cpu/surface.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/toor/forge/forge-master/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o"
	cd /home/toor/forge/forge-master/examples && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o -c /home/toor/forge/forge-master/examples/cpu/surface.cpp

examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.i"
	cd /home/toor/forge/forge-master/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/toor/forge/forge-master/examples/cpu/surface.cpp > CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.i

examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.s"
	cd /home/toor/forge/forge-master/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/toor/forge/forge-master/examples/cpu/surface.cpp -o CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.s

examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o.requires:
.PHONY : examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o.requires

examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o.provides: examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/example_surface_cpu.dir/build.make examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o.provides.build
.PHONY : examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o.provides

examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o.provides.build: examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o

# Object files for target example_surface_cpu
example_surface_cpu_OBJECTS = \
"CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o"

# External object files for target example_surface_cpu
example_surface_cpu_EXTERNAL_OBJECTS =

examples/cpu/surface_cpu: examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o
examples/cpu/surface_cpu: examples/CMakeFiles/example_surface_cpu.dir/build.make
examples/cpu/surface_cpu: /usr/lib64/libGLEWmx.so
examples/cpu/surface_cpu: /usr/lib/x86_64-linux-gnu/libGL.so
examples/cpu/surface_cpu: src/libforge.so
examples/cpu/surface_cpu: examples/CMakeFiles/example_surface_cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable cpu/surface_cpu"
	cd /home/toor/forge/forge-master/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_surface_cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/example_surface_cpu.dir/build: examples/cpu/surface_cpu
.PHONY : examples/CMakeFiles/example_surface_cpu.dir/build

examples/CMakeFiles/example_surface_cpu.dir/requires: examples/CMakeFiles/example_surface_cpu.dir/cpu/surface.cpp.o.requires
.PHONY : examples/CMakeFiles/example_surface_cpu.dir/requires

examples/CMakeFiles/example_surface_cpu.dir/clean:
	cd /home/toor/forge/forge-master/examples && $(CMAKE_COMMAND) -P CMakeFiles/example_surface_cpu.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/example_surface_cpu.dir/clean

examples/CMakeFiles/example_surface_cpu.dir/depend:
	cd /home/toor/forge/forge-master && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/toor/forge/forge-master /home/toor/forge/forge-master/examples /home/toor/forge/forge-master /home/toor/forge/forge-master/examples /home/toor/forge/forge-master/examples/CMakeFiles/example_surface_cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/example_surface_cpu.dir/depend

