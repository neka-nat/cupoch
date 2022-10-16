#!/usr/bin/env python3

def print_help():
    print("Usage:")
    print("    > EncodeShader target_file")
    print("      Generate a header file with license information.")
    print("")
    print("    > EncodeShader target_file source_shader_file")
    print("      Append the shader file to the target header file.")


def write_file_header(fd):
    fd.write("// Automatically generated header file for shader.\n")
    fd.write("// See LICENSE.txt for full license statement.\n")
    fd.write("\n")
    fd.write("#pragma once\n")
    fd.write("\n")


def write_string_header(string_name, fd):
    fd.write("namespace cupoch {\n\n")
    fd.write("namespace visualization {\n\n")
    fd.write("namespace glsl {\n\n")
    fd.write(f"const char * const {string_name} = R\"(\n")


def write_string_footer(fd):
    fd.write("\n)\";\n")
    fd.write("\n}  // namespace cupoch::glsl\n")
    fd.write("\n}  // namespace cupoch::visualization\n")
    fd.write("\n}  // namespace cupoch\n")
    fd.write("\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        print_help()

    if len(sys.argv) == 2:
        with open(sys.argv[1], "w") as fd:
            write_file_header(fd)


    if len(sys.argv) >= 3:
        try:
            fd_out = open(sys.argv[1], "a")
        except FileNotFoundError:
            print("Cannot open file %s" % sys.argv[1])
            sys.exit()

        try:
            fd_in = open(sys.argv[2], "r")
        except FileNotFoundError:
            print("Cannot open file %s" % sys.argv[2])
            sys.exit()

        file_in_name = sys.argv[2]
        string_name = file_in_name.split(".")[0].split("/")[-1]

        fd_out.write("// clang-format off\n")
        write_string_header(string_name, fd_out)
        for l in fd_in.readlines():
            fd_out.write(l)
        write_string_footer(fd_out)
        fd_out.write("// clang-format on\n")

        fd_in.close()
        fd_out.close()
