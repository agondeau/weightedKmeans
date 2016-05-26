/** @file cluster.h
 *  @brief Function prototypes for the k-means clustering.
 *
 *  This contains the function prototypes for the k-means
 *  clustering and eventually any macros, constants, or 
 *  global variables.
 *
 *  @author Alexandre Gondeau
 *  @bug No known bugs.
 */

#ifndef _CLUSTER_H
#define _CLUSTER_H

/* -- Includes -- */

/* libc includes. */
#include <stdint.h>
#include <stdbool.h>

/** @brief Contains clusters informations.
 *
 */
typedef struct _cluster
{
    double *centroid; // Centroid dimensions 
    uint64_t nbData; // Number of data in cluster
} cluster;

/** @brief Contains data informations.
 *
 */
typedef struct _data
{
    double *dim; // Data dimensions
    uint32_t clusterID; // Cluster ID of data
} data;

/** @brief Computes the classical version of k-means  
 *         algorithm.
 *  
 *  This will compute the classical version of k-means  
 *  algorithm from 2 clusters to kmax clusters and with 
 *  nbRep random starts for each numbers of clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param kmax The maximum number of clusters 
 *               that will be computed.
 *  @param nbRep The number of random starts for 
 *               each number of clusters.
 *  @return Void.
 */
void CLUSTER_computeKmeans(data *dat, uint64_t n, uint64_t p, uint32_t kmax,uint32_t nbRep);

/** @brief Computes the weighted (features/objects) 
 *         version of k-means algorithm.
 *
 *  This will compute the weighted (features/objects) 
 *  version of k-means algorithm from 2 clusters to 
 *  kmax clusters and with nbRep random starts for 
 *  each numbers of clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param kmax The maximum number of clusters 
 *               that will be computed.
 *  @param nbRep The number of random starts for 
 *               each number of clusters.
 *  @param internalFeatureWeights The boolean that 
 *               specified if the features weights come 
 *               from internal computation or from a file.
 *  @param featureWeightsFile The string to the features 
 *               weights file.
 *  @param internalObjectWeights The boolean that 
 *               specified if the objects weights come 
 *               from internal computation or from a file.
 *  @param objectWeightsFile The string to the objects 
 *               weights file.
 *  @return Void.
 */
void CLUSTER_computeWeightedKmeans(data *dat, uint64_t n, uint64_t p, uint32_t kmax,uint32_t nbRep, bool internalFeatureWeights, const char *featureWeightsFile, bool internalObjectWeights, const char *objectWeightsFile);

#endif /* _CLUSTER_H */
