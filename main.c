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

/* time header file. */
#include <time.h>

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
bool featuresWeightedKmeans = false; // Computes features weighted k-means or not
bool objectsWeightedKmeans = false; // Computes objects weighted k-means or not
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

    // Init randomness
    srand(time(NULL));

    // Project Informations
    COS("k-means : objects/features weighted implementation");
    COS("Author : Alexandre Gondeau");

    WRN("");
    WRN("Input infomations ----------------------"); 

    if(argc > 1)
    {
        uint8_t i;
        char dataFileName[FILENAME_SIZE_MAX], objectWeightFileName[FILENAME_SIZE_MAX], featureWeightFileName[FILENAME_SIZE_MAX];
        uint32_t nbRep = 1, kmax = 2;
        eMethodType objWeiMet = METHOD_SILHOUETTE; // Objects weights claculation method 
        eMethodType feaWeiMet = METHOD_DISPERSION; // Features weights claculation method

        // Arguments handling
        for(i=1;i<argc;i++)
        {
            if(!strcmp("-i", argv[i]))
            {
                if(argv[i+1] != NULL && strcmp("-r", argv[i+1]) && strcmp("-k", argv[i+1]) && strcmp("-o", argv[i+1]) && strcmp("-f", argv[i+1]) && strcmp("-h", argv[i+1]))
                {
                    strncpy(dataFileName, argv[i+1], FILENAME_SIZE_MAX);
                    INF("Data file : %s", dataFileName);
                    i++;
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
                if(argv[i+1] != NULL && strcmp("-i", argv[i+1]) && strcmp("-k", argv[i+1]) && strcmp("-o", argv[i+1]) && strcmp("-f", argv[i+1]) && strcmp("-h", argv[i+1]))
                {
                    nbRep = atoi(argv[i+1]);
                    INF("Number of replicates : %d", nbRep);    
                    i++;
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
                if(argv[i+1] != NULL && strcmp("-i", argv[i+1]) && strcmp("-r", argv[i+1]) && strcmp("-o", argv[i+1]) && strcmp("-f", argv[i+1]) && strcmp("-h", argv[i+1]))
                {
                    kmax = atoi(argv[i+1]);
                    INF("Number max of clusters : %d", kmax);    
                    i++;
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
                // Set boolean variable for objects weighted k-means
                objectsWeightedKmeans = true;

                if(argv[i+1] != NULL && strcmp("-i", argv[i+1]) && strcmp("-r", argv[i+1]) && strcmp("-k", argv[i+1]) && strcmp("-f", argv[i+1]) && strcmp("-h", argv[i+1]))
                {
                    if(!strcmp("SIL", argv[i+1]))
                    {
                        INF("Object weights : internal computation via silhouette method.");
                        internalObjectWeights = true;
                        objWeiMet = METHOD_SILHOUETTE;
                    }
                    else if(!strcmp("SIL_NK", argv[i+1]))
                    {
                        INF("Object weights : internal computation via silhouette method. The sum of objects weights in a cluster is equal to the number of objects in the cluster.");
                        internalObjectWeights = true;
                        objWeiMet = METHOD_SILHOUETTE_NK;
                    }
                    else if(!strcmp("MED", argv[i+1]))
                    {
                        INF("Object weights : internal computation via median method.");
                        internalObjectWeights = true;
                        objWeiMet = METHOD_MEDIAN;
                    }
                    else if(!strcmp("MED_NK", argv[i+1]))
                    {
                        INF("Object weights : internal computation via median method. The sum of objects weights in a cluster is equal to the number of objects in the cluster.");
                        internalObjectWeights = true;
                        objWeiMet = METHOD_MEDIAN_NK;
                    }
                    else if(!strcmp("MIN_CEN_DIST", argv[i+1]))
                    {
                        INF("Object weights : internal computation via minimum distance to the nearest centroid method.");
                        internalObjectWeights = true;
                        objWeiMet = METHOD_MIN_DIST_CENTROID;
                    }
                    else if(!strcmp("MIN_CEN_DIST_NK", argv[i+1]))
                    {
                        INF("Object weights : internal computation via minimum distance to the nearest centroid method.");
                        internalObjectWeights = true;
                        objWeiMet = METHOD_MIN_DIST_CENTROID_NK;
                    }
                    else if(!strcmp("SUM_DIST_CEN", argv[i+1]))
                    {
                        INF("Object weights : internal computation via the sum of distances with the other centroids method.");
                        internalObjectWeights = true;
                        objWeiMet = METHOD_SUM_DIST_CENTROID;
                    }
                    else
                    {
                        strncpy(objectWeightFileName, argv[i+1], FILENAME_SIZE_MAX);
                        INF("Object weights file : %s", objectWeightFileName);
                    }
                    i++;
                }
                else
                {
                    INF("Objects weights : internal computation (default method silhouette)");
                    internalObjectWeights = true;
                }
            }
            else if(!strcmp("-f", argv[i]))
            {
                // Set boolean variable for features weighted k-means
                featuresWeightedKmeans = true;

                if(argv[i+1] != NULL && strcmp("-i", argv[i+1]) && strcmp("-r", argv[i+1]) && strcmp("-k", argv[i+1]) && strcmp("-o", argv[i+1]) && strcmp("-h", argv[i+1]))
                {
                    if(!strcmp("DISP", argv[i+1]))
                    {
                        INF("Features weights : internal computation via dispersion method.");
                        internalFeatureWeights = true;
                        feaWeiMet = METHOD_DISPERSION;
                    }
                    else
                    {
                        strncpy(featureWeightFileName, argv[i+1], FILENAME_SIZE_MAX);
                        INF("Features weights file : %s", featureWeightFileName);
                    }
                    i++;
                }
                else
                {
                    INF("Features weights : internal computation (default method : dispersion)");
                    internalFeatureWeights = true;
                }
            }
            else if(!strcmp("-h", argv[i]))
            {
                usage();
                return EXIT_SUCCESS;
            }
            else
            {
                ERR("Unknown arg");
                usage();
                return EXIT_FAILURE;
            }
        }

        WRN("");
        WRN("Algorithm informations -----------------");
        uint64_t n, p;
        
        // Read data from input file
        data *dat = readDataFile(dataFileName, &n, &p);

        // Check parameters
        if(kmax >= n)
        {
            ERR("Failure : kmax needs to be < n");
            return EXIT_FAILURE;
        }

        if(dat != NULL)
        {
            // Features weighted k-means
            if(featuresWeightedKmeans && !objectsWeightedKmeans)
            {
                INF("Features weighted k-means");
                CLUSTER_computeFeaturesWeightedKmeans(dat, n, p, kmax, nbRep, internalFeatureWeights, featureWeightFileName, feaWeiMet);
            }
            // Objects weighted k-means or objects and features weighted k-means
            else if((!featuresWeightedKmeans && objectsWeightedKmeans) || (featuresWeightedKmeans && objectsWeightedKmeans))
            {
                INF("Objects (and features) weighted k-means");
                CLUSTER_computeWeightedKmeans3(dat, n, p, kmax, nbRep, internalFeatureWeights, featureWeightFileName, feaWeiMet, internalObjectWeights, objectWeightFileName, objWeiMet); // ow[n]
            }
            // Classical k-means
            else
            {
                INF("Classic k-means");
                CLUSTER_computeKmeans4(dat, n, p, kmax, nbRep);
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
                    dat[i].ind = i;
                    dat[i].clusterID = -1;
                    dat[i].pred = NULL;
                    dat[i].succ = NULL;
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
    WRN("Usage : kmeans -i inputDataFile [-r nbReplicates] [-k clusterMax] [-o objectsWeightsFile/objectsWeightsMethod] [-f featuresWeightsFile/featuresWeightsMethod]");
    WRN("");
    WRN("-i : Specify the input file containing the header (number of data + tab + number of variables) and the data tab spaced.");
    WRN("-r : Specify the number of random starts.");
    WRN("-k : Specify the number max of clusters. K-means will be computed for 2 to clusterMax groups.");
    WRN("-o : Specify the usage of objects weights. If no file containing a priori weights is specified, then the weights are computed internally. The internal computation methods can be selected with the following keys :");
    WRN("       - SIL : based on the silhouette index (default).");
    WRN("       - SIL_NK : based on the silhouette index. The sum of objects weights in a cluster is equal to the number of objects in the cluster.");
    WRN("       - MED : based on the median distance between the object and its partition centroid.");
    WRN("       - MED_NK : based on the median distance between the object and its partition centroid. The sum of objects weights in a cluster is equal to the number of objects in the cluster.");
    WRN("       - MIN_CEN_DIST : based on the minimum euclidean distance between the object and the nearest centroid (different of its own).");
    WRN("       - MIN_CEN_DIST_NK : based on the minimum euclidean distance between the object and the nearest centroid (different of its own). The sum of objects weights in a cluster is equal to the number of objects in the cluster.");
    WRN("       - SUM_DIST_CEN : based on the sum of euclidean distances between the objects and the others centroids.");
    WRN("-f : Specify the usage of features weights. If no file containing a priori weights is specified, then the weights are computed internally. The internal computation methods can be selected with the following keys :");
    WRN("       - DISP : based on a dispersion measure (default).");
}
