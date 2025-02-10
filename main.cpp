// Include necessary headers
#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <chrono>
#include "mpi.h"

// Function to read data from a file into a vector
void read_vector(const std::string &path, std::vector<float> &vec)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << path << std::endl;
        return;
    }

    float input;
    while (file >> input)
    {
        vec.push_back(input);
    }

    file.close();
}

int main(int argc, char **argv)
{
    // Initialize timing mechanism to measure performance
    auto start = std::chrono::high_resolution_clock::now();
    int nrproc, my_rank;

    // MPI initialization and rank identification
    MPI_Init(&argc, &argv);
    MPI_Comm worldCommunicator = MPI_COMM_WORLD;
    MPI_Comm_size(worldCommunicator, &nrproc);
    MPI_Comm_rank(worldCommunicator, &my_rank);

    // Determine the number of processes and divide into two groups
    int subGroupSize = nrproc / 2;
    MPI_Comm subGroupCommunicator;
    MPI_Group world_group, subGroup;

    // Create an array of ranks for creating groups
    std::vector<int> allRanks(nrproc);
    std::iota(allRanks.begin(), allRanks.end(), 0);

    MPI_Comm_group(worldCommunicator, &world_group);

    // Create two subgroups, one for each half of the processes
    if (my_rank < subGroupSize)  
        MPI_Group_incl(world_group, subGroupSize, allRanks.data(), &subGroup);
    else
        MPI_Group_incl(world_group, subGroupSize, allRanks.data() + subGroupSize, &subGroup);

    MPI_Comm_create(worldCommunicator, subGroup, &subGroupCommunicator);
    int my_new_rank;
    MPI_Group_rank(subGroup, &my_new_rank);

    // Distribute data among processes
    int vectorSize;
    std::vector<float> x, y, matrix;
    std::vector<float> x_loc, y_loc, a_loc;

    // Read vectors only if in the respective subgroup
    if (my_new_rank == 0)
    {
        read_vector("x.dat", x);
        read_vector("y.dat", y);
        if (my_rank >= subGroupSize)
            read_vector("mat.dat", matrix);
        vectorSize = x.size();
    }

    // Broadcast the size of vectors to all processes in the subgroup
    MPI_Bcast(&vectorSize, 1, MPI_INT, 0, subGroupCommunicator);

    // Calculate distribution of data elements to processes
    int base_count = vectorSize / subGroupSize;
    int extra_items = vectorSize % subGroupSize;
    std::vector<int> send_counts(nrproc), displacements(nrproc);

    int current_displacement = 0;
    for (int i = 0; i < nrproc; i++)
    {
        send_counts[i] = base_count + (i < extra_items ? 1 : 0);
        displacements[i] = current_displacement;
        current_displacement += send_counts[i];
    }

    // Resize local vectors to receive scattered parts of x and y
    x_loc.resize(send_counts[my_new_rank]);
    y_loc.resize(send_counts[my_new_rank]);
    MPI_Scatterv(x.data(), send_counts.data(), displacements.data(), MPI_FLOAT,
                 x_loc.data(), x_loc.size(), MPI_FLOAT, 0, subGroupCommunicator);

    // Conditionally broadcast y vector and scatter matrix
    if (my_rank >= subGroupSize)
    {
        y.resize(vectorSize);
        MPI_Bcast(y.data(), y.size(), MPI_FLOAT, 0, subGroupCommunicator);

        for (int i = 0; i < subGroupSize; i++)
        {
            send_counts[i] *= vectorSize;
            displacements[i] *= vectorSize;
        }
        a_loc.resize(send_counts[my_new_rank]);
        MPI_Scatterv(matrix.data(), send_counts.data(), displacements.data(), MPI_FLOAT,
                     a_loc.data(), send_counts[my_new_rank], MPI_FLOAT, 0, subGroupCommunicator);
    }
    else
        MPI_Scatterv(y.data(), send_counts.data(), displacements.data(), MPI_FLOAT,
                     y_loc.data(), send_counts[my_new_rank], MPI_FLOAT, 0, subGroupCommunicator);

    // Compute the dot product locally on each process
    float localResult = 0.0;
    for (int i = 0; i < x_loc.size(); i++)
    {
        if (my_rank >= subGroupSize)
        {
            for (int j = 0; j < vectorSize; j++)
                y_loc[i] += y[j] * a_loc[i * vectorSize + j];
        }
        localResult += x_loc[i] * y_loc[i];
    }

    // Reduce results to a single value
    float globalResult = 0.0;
    MPI_Reduce(&localResult, &globalResult, 1, MPI_FLOAT, MPI_SUM, 0, subGroupCommunicator);

    // Handle result communication between subgroups
    if (my_rank >= subGroupSize && my_new_rank == 0)
        MPI_Send(&globalResult, 1, MPI_FLOAT, 0, 0, worldCommunicator);
    else if (my_rank == 0)
    {
        float otherResult;
        MPI_Recv(&otherResult, 1, MPI_FLOAT, subGroupSize, 0, worldCommunicator, MPI_STATUS_IGNORE);

        // Output the final computation and timing information
        std::ofstream outFile("result.txt");

        if (outFile.is_open())
        {
            // Write the computation result to file
            outFile << otherResult / globalResult << std::endl;
            
            // Write the elapsed time

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            outFile << "Elapsed time: " << elapsed.count() << " s\n";
            outFile.close();
        }
        else
        {
            std::cerr << "Unable to open file for writing." << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}
