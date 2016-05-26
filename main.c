/** @file main.c
 *  @brief The kmeans program entrypoint.
 *
 *  This file contains the kmeans's
 *  main() function.
 *
 *  @author Alexandre Gondeau
 *  @bug No known bugs.
 */

/* -- Includes -- */

/* libc includes. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

/* string header file. */
#include <string.h>

/* log include. */
#include "log.h"

/* cluster header file. */
#include "cluster.h"

/* -- Defines -- */

/* main constant defines */
#define FILENAME_SIZE_MAX 256

/* -- Global variables -- */
bool weightedKmeans = false; // Computes weighted k-means or not
bool internalFeatureWeights = false; // The boolean that specified if the features weights come from internal computation or from a file.
bool internalObjectWeights = false; // The boolean that specified if the objects weights come from internal computation or from a file.

/** @brief Reads data from a file and allocates memory
 *         for the data.  
 *
 *  @param fileName The string of the data file to read.
 *  @param n The pointer to the number of the data.
 *  @param p The pointer to the number of data dimensions. 
 *  @return The pointer to the data.
 */
data* readDataFile(const char *fileName, uint64_t *n, uint64_t *p);

/** @brief Frees the allocated memory for the data.  
 *
 *  @param dat The double pointer to the data.
 *  @param n The pointer to the number of the data.
 *  @return Void.
 */
void freeData(data **dat, uint64_t n);

/** @brief Displays the program usage.  
 *
 *  @return Void.
 */
void usage(void);

/** @brief Kmeans entrypoint.
 *  
 *  This is the entrypoint for the kmeans program.
 *
 * @return The program state (SUCCESS/FAILURE).
 */
int main(int argc, char *argv[]) {

    // Project Informations
    COS("k-means : object weighted/feature weighted implementation");
    COS("Author : Alexandre Gondeau");

    WRN("");
    WRN("Input infomations ----------------------"); 

    if(argc > 1)
    {
        uint8_t i;
        char dataFileName[FILENAME_SIZE_MAX], objectWeightFileName[FILENAME_SIZE_MAX], featureWeightFileName[FILENAME_SIZE_MAX];
        uint32_t nbRep, kmax;

        // Arguments handling
        for(i=1;i<argc;i++)
        {
            if(!strcmp("-i", argv[i]))
            {
                if(argv[i+1] != NULL && strcmp("-r", argv[i+1]) && strcmp("-k", argv[i+1]) && strcmp("-o", argv[i+1]) && strcmp("-f", argv[i+1]))
                {
                    strncpy(dataFileName, argv[i+1], FILENAME_SIZE_MAX);
                    INF("Data file : %s", dataFileName);
                }
                else
                {
                    ERR("Bad args");
                    usage();
                    return EXIT_FAILURE;
                }
            }
            else if(!strcmp("-r", argv[i]))
            {
                if(argv[i+1] != NULL && strcmp("-i", argv[i+1]) && strcmp("-k", argv[i+1]) && strcmp("-o", argv[i+1]) && strcmp("-f", argv[i+1]))
                {
                    nbRep = atoi(argv[i+1]);
                    INF("Number of replicates : %d", nbRep);    
                }
                else
                {
                    ERR("Bad args");
                    usage();
                    return EXIT_FAILURE;
                }
            }
            else if(!strcmp("-k", argv[i]))
            {
                if(argv[i+1] != NULL && strcmp("-i", argv[i+1]) && strcmp("-r", argv[i+1]) && strcmp("-o", argv[i+1]) && strcmp("-f", argv[i+1]))
                {
                    kmax = atoi(argv[i+1]);
                    INF("Number max of clusters : %d", kmax);    
                }
                else
                {
                    ERR("Bad args");
                    usage();
                    return EXIT_FAILURE;
                }
            }
            else if(!strcmp("-o", argv[i]))
            {
                // Set boolean variable for weighted k-means
                weightedKmeans = true;

                if(argv[i+1] != NULL && strcmp("-i", argv[i+1]) && strcmp("-r", argv[i+1]) && strcmp("-k", argv[i+1]) && strcmp("-f", argv[i+1]))
                {
                    strncpy(objectWeightFileName, argv[i+1], FILENAME_SIZE_MAX);
                    INF("Object weights file : %s", objectWeightFileName);
                }
                else
                {
                    INF("Object weights : internal computation");
                    internalObjectWeights = true;
                }
            }
            else if(!strcmp("-f", argv[i]))
            {
                // Set boolean variable for weighted k-means
                weightedKmeans = true;

                if(argv[i+1] != NULL && strcmp("-i", argv[i+1]) && strcmp("-r", argv[i+1]) && strcmp("-k", argv[i+1]) && strcmp("-o", argv[i+1]))
                {
                    strncpy(featureWeightFileName, argv[i+1], FILENAME_SIZE_MAX);
                    INF("Features weights file : %s", featureWeightFileName);
                }
                else
                {
                    INF("Feature weights : internal computation");
                    internalFeatureWeights = true;
                }
            }
        }
    
        WRN("");
        WRN("Algorithm informations -----------------");
        uint64_t n, p;
        
        // Read data from input file
        data *dat = readDataFile(dataFileName, &n, &p);
        if(dat != NULL)
        {
            if(weightedKmeans)
            {
                INF("Weighted k-means");
                CLUSTER_computeWeightedKmeans(dat, n, p, kmax, nbRep, internalFeatureWeights, featureWeightFileName, internalObjectWeights, objectWeightFileName);
            }
            else
            {
                INF("Classic k-means");
                CLUSTER_computeKmeans(dat, n, p, kmax, nbRep);
            }

            // Free allocated memory for data
            freeData(&dat, n); 
        }
    }
    else
    {
        usage();
        return EXIT_FAILURE; 
    }

    return EXIT_SUCCESS;
}

data* readDataFile(const char *fileName, uint64_t *n, uint64_t *p)
{
    if(fileName == NULL)
    {
        ERR("Bad parameter");
        return NULL;
    }
    else
    {
        uint64_t i,j;
        data *dat = NULL;
        FILE* fData = fopen(fileName, "r");
        if(fData != NULL)
        {
            fscanf(fData, "%ld %ld",n, p);	//Read matrix parameters fron data file
            INF("n = %ld, p = %ld", *n, *p);

            // Allocate memory for data
            dat = malloc((*n)*sizeof(data));
            if(dat != NULL)
            {
                for(i=0;i<(*n);i++)
                {
                    dat[i].clusterID = -1;
                    dat[i].dim = malloc((*p)*sizeof(double));
                    if(dat[i].dim != NULL)
                    {
                        for(j=0;j<(*p);j++)
                        {
                            fscanf(fData, "%lf", &(dat[i].dim[j]));
                        }
                    }
                    else
                    {
                        ERR("Data dimension memory allocation failed");
                    }
                }
                return dat;
            }
            else
            {
                ERR("Data memory allocation failed");
                return NULL;
            }
        }
        else
        {
            ERR("Reading file failed");
            return NULL;
        }
    }
}

void freeData(data **dat, uint64_t n)
{
    uint64_t i;
    data* d = *dat;
    for(i=0;i<n;i++)
    {
        free(d[i].dim);
    }
    free(d);
}

void usage(void)
{
    WRN("Usage : kmeans -i inputDataFile [-r nbReplicates] [-k clusterMax] [-o objectWeightFile] [-f featureWeightFile]");
}
