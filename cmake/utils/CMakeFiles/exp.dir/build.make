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
include utils/CMakeFiles/exp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include utils/CMakeFiles/exp.dir/compiler_depend.make

# Include the progress variables for this target.
include utils/CMakeFiles/exp.dir/progress.make

# Include the compile flags for this target's objects.
include utils/CMakeFiles/exp.dir/flags.make

utils/CMakeFiles/exp.dir/exp.c.o: utils/CMakeFiles/exp.dir/flags.make
utils/CMakeFiles/exp.dir/exp.c.o: ../utils/exp.c
utils/CMakeFiles/exp.dir/exp.c.o: utils/CMakeFiles/exp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/willmore/vacuumms/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object utils/CMakeFiles/exp.dir/exp.c.o"
	cd /home/willmore/vacuumms/cmake/utils && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT utils/CMakeFiles/exp.dir/exp.c.o -MF CMakeFiles/exp.dir/exp.c.o.d -o CMakeFiles/exp.dir/exp.c.o -c /home/willmore/vacuumms/utils/exp.c

utils/CMakeFiles/exp.dir/exp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/exp.dir/exp.c.i"
	cd /home/willmore/vacuumms/cmake/utils && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/willmore/vacuumms/utils/exp.c > CMakeFiles/exp.dir/exp.c.i

utils/CMakeFiles/exp.dir/exp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/exp.dir/exp.c.s"
	cd /home/willmore/vacuumms/cmake/utils && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/willmore/vacuumms/utils/exp.c -o CMakeFiles/exp.dir/exp.c.s

# Object files for target exp
exp_OBJECTS = \
"CMakeFiles/exp.dir/exp.c.o"

# External object files for target exp
exp_EXTERNAL_OBJECTS =

utils/exp: utils/CMakeFiles/exp.dir/exp.c.o
utils/exp: utils/CMakeFiles/exp.dir/build.make
utils/exp: libraries/libftw_general.so
utils/exp: utils/CMakeFiles/exp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/willmore/vacuumms/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable exp"
	cd /home/willmore/vacuumms/cmake/utils && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/exp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
utils/CMakeFiles/exp.dir/build: utils/exp
.PHONY : utils/CMakeFiles/exp.dir/build

utils/CMakeFiles/exp.dir/clean:
	cd /home/willmore/vacuumms/cmake/utils && $(CMAKE_COMMAND) -P CMakeFiles/exp.dir/cmake_clean.cmake
.PHONY : utils/CMakeFiles/exp.dir/clean

utils/CMakeFiles/exp.dir/depend:
	cd /home/willmore/vacuumms/cmake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/willmore/vacuumms /home/willmore/vacuumms/utils /home/willmore/vacuumms/cmake /home/willmore/vacuumms/cmake/utils /home/willmore/vacuumms/cmake/utils/CMakeFiles/exp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : utils/CMakeFiles/exp.dir/depend
