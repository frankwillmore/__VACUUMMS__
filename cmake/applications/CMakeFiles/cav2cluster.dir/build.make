# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/willmore/spack/opt/spack/linux-ubuntu20.04-graviton/gcc-9.3.0/cmake-3.21.4-ot3yh4ecq3gtb2mwfnoaqnhuwlrsmgci/bin/cmake

# The command to remove a file.
RM = /home/willmore/spack/opt/spack/linux-ubuntu20.04-graviton/gcc-9.3.0/cmake-3.21.4-ot3yh4ecq3gtb2mwfnoaqnhuwlrsmgci/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/willmore/vacuumms

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/willmore/vacuumms/cmake

# Include any dependencies generated for this target.
include applications/CMakeFiles/cav2cluster.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include applications/CMakeFiles/cav2cluster.dir/compiler_depend.make

# Include the progress variables for this target.
include applications/CMakeFiles/cav2cluster.dir/progress.make

# Include the compile flags for this target's objects.
include applications/CMakeFiles/cav2cluster.dir/flags.make

applications/CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.o: applications/CMakeFiles/cav2cluster.dir/flags.make
applications/CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.o: ../applications/cav2cluster/cav2cluster.c
applications/CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.o: applications/CMakeFiles/cav2cluster.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/willmore/vacuumms/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object applications/CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.o"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT applications/CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.o -MF CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.o.d -o CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.o -c /home/willmore/vacuumms/applications/cav2cluster/cav2cluster.c

applications/CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.i"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/willmore/vacuumms/applications/cav2cluster/cav2cluster.c > CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.i

applications/CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.s"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/willmore/vacuumms/applications/cav2cluster/cav2cluster.c -o CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.s

# Object files for target cav2cluster
cav2cluster_OBJECTS = \
"CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.o"

# External object files for target cav2cluster
cav2cluster_EXTERNAL_OBJECTS =

applications/cav2cluster: applications/CMakeFiles/cav2cluster.dir/cav2cluster/cav2cluster.c.o
applications/cav2cluster: applications/CMakeFiles/cav2cluster.dir/build.make
applications/cav2cluster: libraries/libftw_general.so
applications/cav2cluster: applications/CMakeFiles/cav2cluster.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/willmore/vacuumms/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable cav2cluster"
	cd /home/willmore/vacuumms/cmake/applications && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cav2cluster.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
applications/CMakeFiles/cav2cluster.dir/build: applications/cav2cluster
.PHONY : applications/CMakeFiles/cav2cluster.dir/build

applications/CMakeFiles/cav2cluster.dir/clean:
	cd /home/willmore/vacuumms/cmake/applications && $(CMAKE_COMMAND) -P CMakeFiles/cav2cluster.dir/cmake_clean.cmake
.PHONY : applications/CMakeFiles/cav2cluster.dir/clean

applications/CMakeFiles/cav2cluster.dir/depend:
	cd /home/willmore/vacuumms/cmake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/willmore/vacuumms /home/willmore/vacuumms/applications /home/willmore/vacuumms/cmake /home/willmore/vacuumms/cmake/applications /home/willmore/vacuumms/cmake/applications/CMakeFiles/cav2cluster.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : applications/CMakeFiles/cav2cluster.dir/depend
