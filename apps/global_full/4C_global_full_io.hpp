// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GLOBAL_FULL_IO_HPP
#define FOUR_C_GLOBAL_FULL_IO_HPP

#include "4C_config.hpp"

#include "4C_comm_utils.hpp"
#include "4C_io_input_file.hpp"

#include <CLI/CLI.hpp>
FOUR_C_NAMESPACE_OPEN

/**
 * \brief Initializes the input file for reading.
 * \note Currently, this function is a wrapper around Global::set_up_input_file to keep the main
 * routine separate from the Global namespace. In the future, the input file will be set up based on
 * the activated modules.
 */
Core::IO::InputFile setup_input_file(MPI_Comm comm);

/**
 * \brief Emits general metadata to the YAML root reference.
 * \note Currently, this function is a wrapper around Global::emit_general_metadata to keep the main
 * routine separate from the Global namespace. In the future, this function will emit metadata based
 * on the activated modules.
 */
void emit_general_metadata(const Core::IO::YamlNodeRef& root_ref);

/**
 * \brief Structure to hold command line arguments.
 */
struct CommandlineArguments
{
  bool help = false;
  int n_groups = 1;
  bool parameters = false;
  std::vector<int> group_layout = {};
  Core::Communication::NestedParallelismType nptype =
      Core::Communication::NestedParallelismType::no_nested_parallelism;
  int diffgroup = -1;
  int restart = 0;
  std::string restart_file_identifier = "";
  std::vector<int> restart_per_group = {};
  std::vector<std::string> restart_identifier_per_group = {};
  bool interactive = false;
  std::vector<std::pair<std::filesystem::path, std::string>> io_pairs;
  std::filesystem::path input_file_name = "";
  std::string output_file_identifier = "";
};

/**
 * \brief Sets up the Global::Problem instance and puts all the parameters from the input file
 * there.
 */
void setup_global_problem(Core::IO::InputFile& input_file, const CommandlineArguments& arguments,
    const Core::Communication::Communicators& communicators);

/**
 * \brief Writes the hierarchical StackedTimer and flat TimeMonitor information to std::cout
 */
void write_timemonitor(MPI_Comm comm);

/**
 * \brief Exports timing information to a YAML file.
 * \param filename The name of the file to export timings to.
 * \param comm The MPI communicator.
 */
void export_timings(const std::filesystem::path& filename, MPI_Comm comm);

/**
 * \brief Build canonical input/output pairs from positional command line arguments.
 * Due to the legacy argument structure, we separate between the primary input and output (first two
 * positional arguments) and the rest (io_pairs). The latter are only required when using nested
 * parallelism with separate input files.
 * \param io_pairs Vector of strings from the command line representing input/output pairs.
 * \param primary_input The primary input file name (first positional argument).
 * \param primary_output The primary output file identifier (second positional argument).
 * \return A vector of pairs of input file paths and output file identifiers.
 */
std::vector<std::pair<std::filesystem::path, std::string>> build_io_pairs(
    const std::vector<std::string>& io_pairs, const std::filesystem::path& primary_input,
    const std::string& primary_output);

/**
 * \brief Validates cross-compatibility of command line options.
 * \param arguments The parsed command line arguments.
 */
void validate_argument_cross_compatibility(const CommandlineArguments& arguments);
/**
 * \brief Updates input/output identifiers based on group id and nested parallelism type.
 * \param arguments The command line arguments to update.
 * \param group The group id of the current process.
 */
void update_io_identifiers(CommandlineArguments& arguments, int group);

/**
 * \brief Ensures a valid group layout exists.
 *
 * If group_layout is empty and n_groups > 1, computes an equal-size layout based
 * on the size of MPI_COMM_WORLD. Aborts if the processor count is not divisible by the number of
 * groups.
 * \param n_groups Requested number of groups.
 * \param group_layout In/out container for processors per group; populated when initially empty.
 */
void assign_group_layout(const int& n_groups, std::vector<int>& group_layout);

// Custom CLI11 formatter to add extra spacing between options
class SpacedFormatter : public CLI::Formatter
{
 public:
  SpacedFormatter() : Formatter() {}

  std::string make_option(const CLI::Option* opt, bool in_sub) const override
  {
    std::string s = Formatter::make_option(opt, in_sub);
    if (!s.empty() && s.back() == '\n')
      s += '\n';
    else
      s += "\n\n";
    return s;
  }
};

FOUR_C_NAMESPACE_CLOSE

#endif
