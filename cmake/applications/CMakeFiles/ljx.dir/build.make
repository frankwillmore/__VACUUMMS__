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
include applications/CMakeFiles/ljx.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include applications/CMakeFiles/ljx.dir/compiler_depend.make

# Include the progress variables for this target.
include applications/CMakeFiles/ljx.dir/progress.make

# Include the compile flags for this target's objects.
include applications/CMakeFiles/ljx.dir/flags.make

applications/CMakeFiles/ljx.dir/ljx/ljx_main.c.o: applications/CMakeFiles/ljx.dir/flags.make
applications/CMakeFiles/ljx.dir/ljx/ljx_main.c.o: ../applications/ljx/ljx_main.c
applications/CMakeFiles/ljx.dir/ljx/ljx_main.c.o: applications/CMakeFiles/ljx.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/willmore/vacuumms/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object applications/CMakeFiles/ljx.dir/ljx/ljx_main.c.o"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT applications/CMakeFiles/ljx.dir/ljx/ljx_main.c.o -MF CMakeFiles/ljx.dir/ljx/ljx_main.c.o.d -o CMakeFiles/ljx.dir/ljx/ljx_main.c.o -c /home/willmore/vacuumms/applications/ljx/ljx_main.c

applications/CMakeFiles/ljx.dir/ljx/ljx_main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ljx.dir/ljx/ljx_main.c.i"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/willmore/vacuumms/applications/ljx/ljx_main.c > CMakeFiles/ljx.dir/ljx/ljx_main.c.i

applications/CMakeFiles/ljx.dir/ljx/ljx_main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ljx.dir/ljx/ljx_main.c.s"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/willmore/vacuumms/applications/ljx/ljx_main.c -o CMakeFiles/ljx.dir/ljx/ljx_main.c.s

applications/CMakeFiles/ljx.dir/ljx/command_line_parser.c.o: applications/CMakeFiles/ljx.dir/flags.make
applications/CMakeFiles/ljx.dir/ljx/command_line_parser.c.o: ../applications/ljx/command_line_parser.c
applications/CMakeFiles/ljx.dir/ljx/command_line_parser.c.o: applications/CMakeFiles/ljx.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/willmore/vacuumms/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object applications/CMakeFiles/ljx.dir/ljx/command_line_parser.c.o"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT applications/CMakeFiles/ljx.dir/ljx/command_line_parser.c.o -MF CMakeFiles/ljx.dir/ljx/command_line_parser.c.o.d -o CMakeFiles/ljx.dir/ljx/command_line_parser.c.o -c /home/willmore/vacuumms/applications/ljx/command_line_parser.c

applications/CMakeFiles/ljx.dir/ljx/command_line_parser.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ljx.dir/ljx/command_line_parser.c.i"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/willmore/vacuumms/applications/ljx/command_line_parser.c > CMakeFiles/ljx.dir/ljx/command_line_parser.c.i

applications/CMakeFiles/ljx.dir/ljx/command_line_parser.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ljx.dir/ljx/command_line_parser.c.s"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/willmore/vacuumms/applications/ljx/command_line_parser.c -o CMakeFiles/ljx.dir/ljx/command_line_parser.c.s

applications/CMakeFiles/ljx.dir/ljx/io_setup.c.o: applications/CMakeFiles/ljx.dir/flags.make
applications/CMakeFiles/ljx.dir/ljx/io_setup.c.o: ../applications/ljx/io_setup.c
applications/CMakeFiles/ljx.dir/ljx/io_setup.c.o: applications/CMakeFiles/ljx.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/willmore/vacuumms/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object applications/CMakeFiles/ljx.dir/ljx/io_setup.c.o"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT applications/CMakeFiles/ljx.dir/ljx/io_setup.c.o -MF CMakeFiles/ljx.dir/ljx/io_setup.c.o.d -o CMakeFiles/ljx.dir/ljx/io_setup.c.o -c /home/willmore/vacuumms/applications/ljx/io_setup.c

applications/CMakeFiles/ljx.dir/ljx/io_setup.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ljx.dir/ljx/io_setup.c.i"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/willmore/vacuumms/applications/ljx/io_setup.c > CMakeFiles/ljx.dir/ljx/io_setup.c.i

applications/CMakeFiles/ljx.dir/ljx/io_setup.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ljx.dir/ljx/io_setup.c.s"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/willmore/vacuumms/applications/ljx/io_setup.c -o CMakeFiles/ljx.dir/ljx/io_setup.c.s

applications/CMakeFiles/ljx.dir/ljx/energy.c.o: applications/CMakeFiles/ljx.dir/flags.make
applications/CMakeFiles/ljx.dir/ljx/energy.c.o: ../applications/ljx/energy.c
applications/CMakeFiles/ljx.dir/ljx/energy.c.o: applications/CMakeFiles/ljx.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/willmore/vacuumms/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object applications/CMakeFiles/ljx.dir/ljx/energy.c.o"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT applications/CMakeFiles/ljx.dir/ljx/energy.c.o -MF CMakeFiles/ljx.dir/ljx/energy.c.o.d -o CMakeFiles/ljx.dir/ljx/energy.c.o -c /home/willmore/vacuumms/applications/ljx/energy.c

applications/CMakeFiles/ljx.dir/ljx/energy.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ljx.dir/ljx/energy.c.i"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/willmore/vacuumms/applications/ljx/energy.c > CMakeFiles/ljx.dir/ljx/energy.c.i

applications/CMakeFiles/ljx.dir/ljx/energy.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ljx.dir/ljx/energy.c.s"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/willmore/vacuumms/applications/ljx/energy.c -o CMakeFiles/ljx.dir/ljx/energy.c.s

applications/CMakeFiles/ljx.dir/ljx/graphics.c.o: applications/CMakeFiles/ljx.dir/flags.make
applications/CMakeFiles/ljx.dir/ljx/graphics.c.o: ../applications/ljx/graphics.c
applications/CMakeFiles/ljx.dir/ljx/graphics.c.o: applications/CMakeFiles/ljx.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/willmore/vacuumms/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object applications/CMakeFiles/ljx.dir/ljx/graphics.c.o"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT applications/CMakeFiles/ljx.dir/ljx/graphics.c.o -MF CMakeFiles/ljx.dir/ljx/graphics.c.o.d -o CMakeFiles/ljx.dir/ljx/graphics.c.o -c /home/willmore/vacuumms/applications/ljx/graphics.c

applications/CMakeFiles/ljx.dir/ljx/graphics.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ljx.dir/ljx/graphics.c.i"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/willmore/vacuumms/applications/ljx/graphics.c > CMakeFiles/ljx.dir/ljx/graphics.c.i

applications/CMakeFiles/ljx.dir/ljx/graphics.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ljx.dir/ljx/graphics.c.s"
	cd /home/willmore/vacuumms/cmake/applications && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/willmore/vacuumms/applications/ljx/graphics.c -o CMakeFiles/ljx.dir/ljx/graphics.c.s

# Object files for target ljx
ljx_OBJECTS = \
"CMakeFiles/ljx.dir/ljx/ljx_main.c.o" \
"CMakeFiles/ljx.dir/ljx/command_line_parser.c.o" \
"CMakeFiles/ljx.dir/ljx/io_setup.c.o" \
"CMakeFiles/ljx.dir/ljx/energy.c.o" \
"CMakeFiles/ljx.dir/ljx/graphics.c.o"

# External object files for target ljx
ljx_EXTERNAL_OBJECTS =

applications/ljx: applications/CMakeFiles/ljx.dir/ljx/ljx_main.c.o
applications/ljx: applications/CMakeFiles/ljx.dir/ljx/command_line_parser.c.o
applications/ljx: applications/CMakeFiles/ljx.dir/ljx/io_setup.c.o
applications/ljx: applications/CMakeFiles/ljx.dir/ljx/energy.c.o
applications/ljx: applications/CMakeFiles/ljx.dir/ljx/graphics.c.o
applications/ljx: applications/CMakeFiles/ljx.dir/build.make
applications/ljx: libraries/libftw_general.so
applications/ljx: applications/CMakeFiles/ljx.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/willmore/vacuumms/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking C executable ljx"
	cd /home/willmore/vacuumms/cmake/applications && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ljx.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
applications/CMakeFiles/ljx.dir/build: applications/ljx
.PHONY : applications/CMakeFiles/ljx.dir/build

applications/CMakeFiles/ljx.dir/clean:
	cd /home/willmore/vacuumms/cmake/applications && $(CMAKE_COMMAND) -P CMakeFiles/ljx.dir/cmake_clean.cmake
.PHONY : applications/CMakeFiles/ljx.dir/clean

applications/CMakeFiles/ljx.dir/depend:
	cd /home/willmore/vacuumms/cmake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/willmore/vacuumms /home/willmore/vacuumms/applications /home/willmore/vacuumms/cmake /home/willmore/vacuumms/cmake/applications /home/willmore/vacuumms/cmake/applications/CMakeFiles/ljx.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : applications/CMakeFiles/ljx.dir/depend
