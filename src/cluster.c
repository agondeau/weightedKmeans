/** @file cluster.c
 *  @brief Function declarations for the k-means clustering.
 *
 *  This contains the function declarations for the k-means
 *  clustering and eventually any macros, constants,or 
 *  global variables.
 *
 *  @author Alexandre Gondeau
 *  @bug No known bugs.
 */

/* -- Includes -- */

/* libc includes. */
#include <stdlib.h>

/* time header file. */
#include <time.h>

/* math header file. */
#include <math.h>

/* cluster header file. */
#include "cluster.h"

/* log include. */
#include "log.h"

/* -- Defines -- */

/* Clustering constant defines. */
#define NB_ITER 100
#define K_MIN 2

/* -- Enumerations -- */

/** @brief Contains the different distance calculation types.
 *
 */
typedef enum _eDistanceType 
{
    DISTANCE_EUCLIDEAN = 0,
    DISTANCE_OTHER
} eDistanceType;

/** @brief Contains the different weights calculation types.
 *
 */
typedef enum _eMethodType 
{
    METHOD_SILHOUETTE = 0,
    METHOD_OTHER
} eMethodType; 

/** @brief Proceeds to a fake assigenment of data to  
 *         the different clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static void CLUSTER_fakeDataAssignmentToCentroids(data *dat, uint64_t n, uint32_t k);

/** @brief Allocates memory for the clusters dimensions.  
 *
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static void CLUSTER_initClusters(uint64_t p, cluster *c, uint32_t k);

/** @brief Frees allocated memory for the clusters dimensions.  
 *
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static void CLUSTER_freeClusters(cluster *c, uint32_t k);

/** @brief Chooses random data as centroids.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static void CLUSTER_randomCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Assigns data to the nearest centroid.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static double CLUSTER_assignDataToCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Assigns weighted data to the nearest centroid.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param fw The pointer to the features weights.
 *  @param ow The pointer to the objects weights.
 *  @return Void.
 */
static double CLUSTER_assignWeightedDataToCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *fw, double *ow);

/** @brief Computes the squared distance between a point and a cluster.
 *
 *  @param dat The pointer to the datum.
 *  @param p The number of datum dimensions.
 *  @param c The pointer to the cluster.
 *  @param d The type of distance calculation. 
 *  @return the computed distance between the point and the cluster.
 */
static double CLUSTER_computeSquaredDistancePointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d);

/** @brief Computes the squared distance between a weighted 
 *         point and a cluster.
 *
 *  @param dat The pointer to the datum.
 *  @param p The number of datum dimensions.
 *  @param c The pointer to the cluster.
 *  @param d The type of distance calculation. 
 *  @param fw The pointer to the features weights.
 *  @param ow The point weight.
 *  @return The computed distance between the weigted point 
 *          and the cluster.
 */
static double CLUSTER_computeSquaredDistanceWeightedPointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d, double *fw, double ow);

/** @brief Updates the centroids positions.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static void CLUSTER_computeCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Computes the classical version of k-means  
 *         algorithm for k clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return The sum of squared errors for the clustering.
 */
static double CLUSTER_kmeans(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c);

/** @brief Computes the classical version of k-means  
 *         algorithm for k clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param internalFeatureWeights The boolean that 
 *           specified if the features weights come 
 *           from internal computation or from a file.
 *  @param fw The pointer to the features weights.
 *  @param internalObjectWeights The boolean that 
 *           specified if the objects weights come 
 *           from internal computation or from a file.
 *  @param ow The pointer to the objects weights.
 *  @return The sum of squared errors for the clustering.
 */
static double CLUSTER_weightedKmeans(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, double *fw, bool internalObjectWeights, double *ow);

/** @brief Computes the silhouette score for a 
 *         clustering.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return The computed silhouette score for the clustering.
 */
static double CLUSTER_computeSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Computes the distance between a point
 *         and an other point .
 *
 *  @param iDat The pointer to the first point.
 *  @param jDat The pointer to the second point.
 *  @param p The number of data dimensions.
 *  @param d The type of distance calculation. 
 *  @return The computed distance between the first point 
 *          and the second point.
 */
static double CLUSTER_computeDistancePointToPoint(data *iDat, data *jDat, uint64_t p, eDistanceType d);

/** @brief Computes the total sum of squares for 
 *         the data.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @return The computed total sum of squares for the data.
 */
static double CLUSTER_computeTSS(data *dat, uint64_t n, uint64_t p); 

/** @brief Computes the variance ratio criterion for 
 *         a clustering.
 *
 *  @param TSS The computed total sum of squares for the data.
 *  @param SSE The sum of squared errors for the clustering.
 *  @param n The number of the data.
 *  @param k The number of clusters. 
 *  @return The computed variance ratio criterion for the 
 *          clustering.
 */
static double CLUSTER_computeVRC(double TSS, double SSE, uint64_t n, uint32_t k);  

/** @brief Computes the Calinski-Harabasz criterion for 
 *         a clustering.
 *
 *  @param TSS The computed total sum of squares for the data.
 *  @param SSE The sum of squared errors for the clustering.
 *  @param n The number of the data.
 *  @param k The number of clusters. 
 *  @return The computed Calinski-Harabasz criterion for the 
 *          clustering.
 */
static double CLUSTER_computeCH(double TSS, double SSE, uint64_t n, uint32_t k); 

/** @brief Initializes weights values to 1. 
 *
 *  @param w The pointer to the weights.
 *  @param dim The number of weights dimensions. 
 *  @return Void.
 */
static void CLUSTER_initWeights(double *w, uint64_t dim);

/** @brief Computes objects weights via different
 *         methods.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param ow The pointer to the objects weights.
 *  @param m The method for objects weights calculation.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeights(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow, eMethodType m);

/** @brief Computes objects weights via silhouette
 *         score.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param ow The pointer to the objects weights.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow);

/** @brief Detects and removes noisy points.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param nToRemove The pointer to the number 
 *                   of noisy points to remove.
 *  @param kToRemove The pointer to the number 
 *                   of noisy clusters to remove.
 *  @return Void.
 */
static void CLUSTER_removeNoise(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint64_t *nToRemove, uint32_t *kToRemove);

/** @brief Computes the sum of squared errors for a 
 *         clustering. 
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param fw The pointer to the features weights.
 *  @param ow The pointer to the objects weights.
 *  @return The computed sum of squared errors for a 
 *          clustering.
 */
static double CLUSTER_computeSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *fw, double *ow);


/* -- Function definitions -- */

void CLUSTER_computeKmeans(data *dat, uint64_t n, uint64_t p, uint32_t kmax,uint32_t nbRep)
{
    uint32_t i, k;
    uint64_t j;
    double statSil[kmax], statVRC[kmax], statCH[kmax];
    uint32_t chGrp[kmax+1][n]; // Data membership for each k (CH)

    // Initialize statistics
    for(k=kmax;k>=K_MIN;k--)
    {
        statSil[k] = 0.0;
        statVRC[k] = 0.0;
        statCH[k] = 0.0;
    }

    // Compute total sum of squares 
    double TSS = CLUSTER_computeTSS(dat, n, p);

    for(i=0;i<nbRep;i++) // Number of replicates
    {
        for(k=kmax;k>=K_MIN;k--) // From kMax to kMin
        {
            INF("Compute for k = %d", k);

            cluster c[k];
            // Allocate clusters dimension memory
            CLUSTER_initClusters(p, c, k);

            double SSE = CLUSTER_kmeans(dat, n, p, k, c);

            // Compute silhouette statistic
            double sil = CLUSTER_computeSilhouette(dat, n, p, c, k);
            SAY("Silhouette = %lf", sil);

            // Compute VRC statistic
            SAY("TSS = %lf, SSE = %lf, n = %ld, k = %d", TSS, SSE, n, k);
            double vrc = CLUSTER_computeVRC(TSS, SSE, n, k);
            SAY("VRC = %lf", vrc);

            // Compute CH statistic
            double ch = CLUSTER_computeCH(TSS, SSE, n, k);
            SAY("CH = %lf", ch);

            // Save best silhouette statistic for each k
            if(sil > statSil[k])
                statSil[k] = sil;

            // Save best VRC statistic for each k
            if(vrc > statVRC[k])
                statVRC[k] = vrc;

            // Save best CH statistic for each k
            if(ch > statCH[k])
            {
                statCH[k] = ch;
                // Save data membership for each k (CH)
                for(j=0;j<n;j++)
                {
                    chGrp[k][j] = dat[j].clusterID;
                }
            }

            // Free clusters dimension memory
            CLUSTER_freeClusters(c, k);
        }
    }

    // Retrieve the overall best statistics
    double silMax = 0.0, vrcMax = 0.0, chMax = 0.0;
    uint32_t kSilMax, kVrcMax, kChMax;
    for(k=kmax;k>=K_MIN;k--)
    {
        if(statSil[k] > silMax)
        {
            silMax = statSil[k];
            kSilMax = k;
        }

        if(statVRC[k] > vrcMax)
        {
            vrcMax = statVRC[k];
            kVrcMax = k;
        }

        if(statCH[k] > chMax)
        {
            chMax = statCH[k];
            kChMax = k;
        }
    }

    WRN("");
    WRN("Final statistics -----------------------");
    INF("Best silhouette : %lf for k = %d", silMax, kSilMax);
    INF("Best VRC : %lf for k = %d", vrcMax, kVrcMax);
    INF("Best CH : %lf for k = %d", chMax, kChMax);
    WRN("");
    WRN("Data membership for each k (CH) --------");
    printf("dataId\t");
    for(k=kmax;k>=K_MIN;k--)
        printf("%d-Gr\t", k);
    INF("");
    for(j=0;j<n;j++)
    {
        printf("%ld\t", j);
        for(k=kmax;k>=K_MIN;k--)
            printf("%d\t", chGrp[k][j]);
        INF("");
    }
    WRN("----------------------------------------");
}

void CLUSTER_computeWeightedKmeans(data *dat, uint64_t n, uint64_t p, uint32_t kmax,uint32_t nbRep, bool internalFeatureWeights, const char *featureWeightsFile, bool internalObjectWeights, const char *objectWeightsFile)
{
    uint32_t i, k;
    uint64_t j;
    double statSil[kmax], statVRC[kmax], statCH[kmax];
    double ow[n], fw[p];
    uint32_t chGrp[kmax+1][n]; // Data membership for each k (CH)

    // Initialize statistics
    for(k=kmax;k>=K_MIN;k--)
    {
        statSil[k] = 0.0;
        statVRC[k] = 0.0;
        statCH[k] = 0.0;
    }

    // Initialize weights to 1.0
    CLUSTER_initWeights(fw, p);
    CLUSTER_initWeights(ow, n);

    // Compute total sum of squares 
    double TSS = CLUSTER_computeTSS(dat, n, p);

    if(internalFeatureWeights == false)
    {
        // Read feature weights from file
    }

    if(internalObjectWeights == false)
    {
        // Read object weights from file
    }

    for(i=0;i<nbRep;i++) // Number of replicates
    {
        for(k=kmax;k>=K_MIN;k--) // From kMax to kMin
        {
            if(internalObjectWeights == true)
            {
                // Initialize object weights to 1.0
                CLUSTER_initWeights(ow, n);
            }

            INF("Compute for k = %d", k);

            cluster c[k];
            // Allocate clusters dimension memory
            CLUSTER_initClusters(p, c, k);

            double SSE = CLUSTER_weightedKmeans(dat, n, p, k, c, internalFeatureWeights, fw, internalObjectWeights, ow);

            // Remove noise
            /*uint32_t kToRemove = 0;
            uint64_t nToRemove = 0; 
            CLUSTER_removeNoise(dat, n, p, c, k, &nToRemove, &kToRemove);*/ 

            // Compute silhouette statistic
            double sil = CLUSTER_computeSilhouette(dat, n, p, c, k);
            SAY("Silhouette = %lf", sil);

            // Update SSE after noise deletion
            //SSE = CLUSTER_computeSSE(dat, n, p, c, k, fw, ow);

            // Compute VRC statistic
            SAY("TSS = %lf, SSE = %lf, n = %ld, k = %d", TSS, SSE, n, k);
            double vrc = CLUSTER_computeVRC(TSS, SSE, n, k);
            SAY("VRC = %lf", vrc);

            // Compute CH statistic
            double ch = CLUSTER_computeCH(TSS, SSE, n, k);
            SAY("CH = %lf", ch);

            // Save best silhouette statistic for each k
            if(sil > statSil[k])
            {
                statSil[k] = sil;
                /*for(j=0;j<n;j++)
                {
                    chGrp[k][j] = dat[j].clusterID;
                }*/
            }

            // Save best VRC statistic for each k
            if(vrc > statVRC[k])
                statVRC[k] = vrc;

            // Save best CH statistic for each k
            if(ch > statCH[k])
            {
                statCH[k] = ch;
                // Save data membership for each k (CH)
                for(j=0;j<n;j++)
                {
                    chGrp[k][j] = dat[j].clusterID;
                }
            }

            // Free clusters dimension memory
            CLUSTER_freeClusters(c, k);
        }
    }

    // Retrieve the overall best statistics
    double silMax = 0.0, vrcMax = 0.0, chMax = 0.0;
    uint32_t kSilMax, kVrcMax, kChMax;
    for(k=kmax;k>=K_MIN;k--)
    {
        if(statSil[k] > silMax)
        {
            silMax = statSil[k];
            kSilMax = k;
        }

        if(statVRC[k] > vrcMax)
        {
            vrcMax = statVRC[k];
            kVrcMax = k;
        }

        if(statCH[k] > chMax)
        {
            chMax = statCH[k];
            kChMax = k;
        }
    }

    WRN("");
    WRN("Final statistics -----------------------");
    INF("Best silhouette : %lf for k = %d", silMax, kSilMax);
    INF("Best VRC : %lf for k = %d", vrcMax, kVrcMax);
    INF("Best CH : %lf for k = %d", chMax, kChMax);
    WRN("");
    WRN("Data membership for each best k (CH) --------");
    printf("dataId\t");
    for(k=kmax;k>=K_MIN;k--)
        printf("%d-Gr\t", k);
    INF("");
    for(j=0;j<n;j++)
    {
        printf("%ld\t", j);
        for(k=kmax;k>=K_MIN;k--)
            printf("%d\t", chGrp[k][j]);
        INF("");
    }
    WRN("----------------------------------------");
}

static double CLUSTER_kmeans(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c)
{
    if(dat == NULL || n < 2 || p < 1 || k < 2)
    {
        ERR("k-means : bad parameter : dat = %p, n = %ld, p = %ld, k = %d", dat, n, p, k);
        return -1.0;
    }
    else
    {
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, k);
        CLUSTER_randomCentroids(dat, n, p, c, k);
        CLUSTER_assignDataToCentroids(dat, n, p, c, k);
        SAY("");

        uint8_t iter=0;
        double SSEref=1.0e20, SSE;
        bool conv = false; 
        while(iter < NB_ITER && conv == false)
        {
            CLUSTER_computeCentroids(dat, n, p, c, k);
            SSE = CLUSTER_assignDataToCentroids(dat, n, p, c, k);
            SAY("");

            SAY("SSEref = %lf, SSE = %lf", SSEref, SSE);
            if(fabs(SSEref-SSE)>(SSE/1000.0))	
                SSEref=SSE;
            else
            {
                INF("Has converged in %d iterations", iter+1);            
                conv = true;
            }

            iter++;
        }

        return SSE; 
    }
}

static double CLUSTER_weightedKmeans(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, double *fw, bool internalObjectWeights, double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || k < 2 || fw == NULL || ow == NULL)
    {
        ERR("k-means : bad parameter : dat = %p, n = %ld, p = %ld, k = %d", dat, n, p, k);
        return -1.0;
    }
    else
    {
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, k);
        CLUSTER_randomCentroids(dat, n, p, c, k);
        CLUSTER_assignWeightedDataToCentroids(dat, n, p, c, k, fw, ow);
        SAY("");

        if(internalObjectWeights == true)
        {
            // Calculate object weights
            CLUSTER_computeObjectWeights(dat, n, p, c, k, ow, METHOD_SILHOUETTE);
        }

        uint8_t iter = 0;
        double SSEref = 1.0e20, SSE;
        bool conv = false; 
        while(iter < NB_ITER && conv == false)
        {
            CLUSTER_computeCentroids(dat, n, p, c, k);
            SSE = CLUSTER_assignWeightedDataToCentroids(dat, n, p, c, k, fw, ow);
            SAY("");

            if(internalObjectWeights == true)
            {
                // Update object weights
                CLUSTER_computeObjectWeights(dat, n, p, c, k, ow, METHOD_SILHOUETTE);
            }

            SAY("SSEref = %lf, SSE = %lf", SSEref, SSE);
            if(fabs(SSEref-SSE)>(SSE/1000.0))	
                SSEref=SSE;
            else
            {
                INF("Has converged in %d iterations", iter+1);            
                conv = true;
            }

            iter++;
        }

        return SSE; 
    }
}

static void CLUSTER_fakeDataAssignmentToCentroids(data *dat, uint64_t n, uint32_t k)
{
    if(dat == NULL || n < 2 || k < 2)
        ERR("Bad parameter");
    else
    {
        uint64_t i;
        uint32_t l = 0;
        for(i=0;i<n;i++)
        {
            // Assign fake cluster to avoid null cluster
            if(l == k)
                l = 0;

            dat[i].clusterID = l;
            l++;
        }
    }
}

static void CLUSTER_initClusters(uint64_t p, cluster *c, uint32_t k)
{
    if(p < 1 || c == NULL || k < 2)
        ERR("Bad parameter");
    else
    {
        uint32_t l;
        for(l=0;l<k;l++)
        {
            c[l].centroid = malloc(p*sizeof(double)); // Allocate cluster dimension memory
            if(c[l].centroid == NULL)
            {
                ERR("Cluster dimension memory allocation failed");
            }
        }
    }
}

static void CLUSTER_freeClusters(cluster *c, uint32_t k)
{
    if(c == NULL || k < 2)
        ERR("Bad parameter");
    else
    {
        uint32_t l;
        for(l=0;l<k;l++)
        {
            free(c[l].centroid); // Free cluster dimension memory
        }
    }
}

static void CLUSTER_randomCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
        ERR("Bad parameter");
    else
    {
        // Init randomness
        srand(time(NULL));

        uint32_t l;
        uint64_t j;

        // Initialize cluster data number
        for(l=0;l<k;l++)
            c[l].nbData = 0;

        // Fake assignment
        uint64_t i;
        for(i=0;i<n;i++)
            c[dat[i].clusterID].nbData++;


        for(l=0;l<k;l++)
        {
            if(c[l].centroid != NULL)
            {
                // Use random data as centroids
                uint64_t randInd = rand() % n;
                for(j=0;j<p;j++)
                    c[l].centroid[j] = dat[randInd].dim[j];

                // Init data cluster number

                //c[l].nbData = 0;
                //SAY("Data[%ld] as cluster", randInd);
                //SAY("Nb data c%d = %ld", l, c[l].nbData);
            }
            else
            {
                ERR("Cluster dimensions not allocated");
            }
        }
    }
}

static double CLUSTER_assignDataToCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i;
        uint32_t l;
        double SSE = 0.0;

        for(i=0;i<n;i++)
        {
            double minDist;
            uint32_t minK;
            for(l=0;l<k;l++)
            {
                // Calculate squared Euclidean distance
                double dist = CLUSTER_computeSquaredDistancePointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN);

                if(l == 0)
                {
                    minDist = dist;
                    minK = l;
                }
                else
                {
                    if(dist < minDist)
                    {
                        minDist = dist;
                        minK = l; // Save the cluster for the min distance
                    }
                }
            }

            if((c[dat[i].clusterID].nbData-1) != 0) // Avoid null cluster
            {
                c[dat[i].clusterID].nbData--; // Decrease previous cluster data number
                dat[i].clusterID = minK; // Assign data to cluster
                c[dat[i].clusterID].nbData++; // Increase new cluster data number
            }

            SSE += minDist;

            SAY("dat[%ld] centroid : %d (nb data = %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);
        }

        for(l=0;l<k;l++)
        {
            SAY("Cluster %d : %ld data", l, c[l].nbData);
        }

        return SSE;
    }
}

static double CLUSTER_assignWeightedDataToCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *fw, double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || fw == NULL || ow == NULL)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i;
        uint32_t l;
        double SSE = 0.0;

        for(i=0;i<n;i++)
        {
            double minDist;
            uint32_t minK;
            for(l=0;l<k;l++)
            {
                // Calculate squared Euclidean distance
                double dist = CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN, fw, ow[i]);

                if(l == 0)
                {
                    minDist = dist;
                    minK = l;
                }
                else
                {
                    if(dist < minDist)
                    {
                        minDist = dist;
                        minK = l; // Save the cluster for the min distance
                    }
                }
            }

            if((c[dat[i].clusterID].nbData-1) != 0) // Avoid null cluster
            {
                c[dat[i].clusterID].nbData--; // Decrease previous cluster data number
                dat[i].clusterID = minK; // Assign data to cluster
                c[dat[i].clusterID].nbData++; // Increase new cluster data number
            }

            SSE += minDist;

            //SAY("dat[%ld] centroid : %d (nb data = %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);
        }

        for(l=0;l<k;l++)
        {
            SAY("Cluster %d : %ld data", l, c[l].nbData);
        }

        return SSE;
    }
}

static double CLUSTER_computeSquaredDistancePointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d)
{
    if(dat == NULL || p < 1 || c == NULL)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double dist = 0;
        switch(d)
        {
            default:
            case DISTANCE_EUCLIDEAN:
                {
                    uint64_t j;
                    for(j=0;j<p;j++)
                        dist += pow((dat->dim[j] - c->centroid[j]), 2.0);
                }
                break;
            case DISTANCE_OTHER:
                {
                    WRN("Not implemented yet");
                    dist = -1;
                }
                break;
        }

        return dist;
    }
}

static double CLUSTER_computeSquaredDistanceWeightedPointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d, double *fw, double ow)
{
    if(dat == NULL || p < 1 || c == NULL || fw == NULL)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double dist = 0;
        switch(d)
        {
            default:
            case DISTANCE_EUCLIDEAN:
                {
                    uint64_t j;
                    for(j=0;j<p;j++)
                        dist += fw[j]*pow((dat->dim[j] - c->centroid[j]), 2.0); // Apply feature weights
                }
                break;
            case DISTANCE_OTHER:
                {
                    WRN("Not implemented yet");
                    dist = -1;
                }
                break;
        }

        return (ow*dist); // Apply object weight
    }
}

static void CLUSTER_computeCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
    }
    else
    {

    uint64_t i,j;
    uint32_t l;

    // Reset each centroid dimension to 0
    for(l=0;l<k;l++)
        for(j=0;j<p;j++)
            c[l].centroid[j] = 0.0;

    // Compute the new centroid dimension
    for(i=0;i<n;i++)
    {
        for(j=0;j<p;j++)
        {
            c[dat[i].clusterID].centroid[j] += (dat[i].dim[j]/(double)c[dat[i].clusterID].nbData);   
        }
    }    
    }
}

static double CLUSTER_computeSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -2.0;
    }
    else
    {
        double a[n], b[n], s[n], sk[k], distCluster[k], dist[n][n];
        uint64_t i,j;
        uint32_t l;

        // Initilize sik
        for(l=0;l<k;l++) 
            sk[l] = 0.0;

        // Create the distance matrix of i vs j
        for(i=0;i<n;i++)
            for (j=0;j<n;j++)
                dist[i][j]= CLUSTER_computeDistancePointToPoint(&(dat[i]), &(dat[j]), p, DISTANCE_EUCLIDEAN);

        for(i=0;i<n;i++)
        {
            // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
            double d = 0.0;
            for(j=0;j<n;j++)
            {
                if(j != i && dat[j].clusterID == dat[i].clusterID)
                    d += dist[i][j];

                if((c[dat[i].clusterID].nbData - 1) == 0)
                    a[i] = 0.0;
                else
                    a[i] = d/(double)(c[dat[i].clusterID].nbData - 1);
            }

            // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
            for(l=0;l<k;l++)
                distCluster[l] = 0.0;

            for(j=0;j<n;j++)
                if(dat[j].clusterID != dat[i].clusterID)
                    distCluster[dat[j].clusterID] += (dist[i][j]/c[dat[j].clusterID].nbData);
            b[i] = 1.0e20;
            for(l=0;l<k;l++)
                if(l != dat[i].clusterID && distCluster[l] != 0 && distCluster[l] < b[i])
                    b[i] = distCluster[l];

            // Calculate s[i]
            s[i] = (b[i] != a[i]) ?  (b[i] - a[i]) / fmax(a[i], b[i]) : 0.0;
            //SAY("i = %ld, s[i] = %lf, b[i] = %lf, a[i] = %lf, dat[i].clusterID = %d, c[dat[i].clusterID].nbData = %ld", i, s[i], b[i], a[i], dat[i].clusterID, c[dat[i].clusterID].nbData);
            sk[dat[i].clusterID]+= s[i] / (double)c[dat[i].clusterID].nbData;
        }

        double sil = 0.0;
        //uint32_t kToRemove = 0;
        for(l=0;l<k;l++)
        {
            SAY("sk[%d] = %lf", l, sk[l]);
            /*if(sk[l] == 0.0)
                kToRemove++;
            else*/
                if (!isnan(sk[l]))
                    sil += sk[l];
        }

        return (sil/k);

        // Handle null cluster
        /*if((k-kToRemove) == 1)
            return (sil/k);
        else
            return (sil/(k-kToRemove));*/
    }
}

static double CLUSTER_computeDistancePointToPoint(data *iDat, data *jDat, uint64_t p, eDistanceType d)
{
    if(iDat == NULL || jDat == NULL || p < 1)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double dist = 0;
        switch(d)
        {
            default:
            case DISTANCE_EUCLIDEAN:
                {
                    uint64_t j;
                    for(j=0;j<p;j++)
                        dist += pow((iDat->dim[j] - jDat->dim[j]), 2.0);
                }
                break;
            case DISTANCE_OTHER:
                {
                    WRN("Not implemented yet");
                    dist = -1;
                }
                break;
        }

        return sqrt(dist);
    }
}

static double CLUSTER_computeTSS(data *dat, uint64_t n, uint64_t p)
{
    if(dat == NULL || n < 2 || p < 1)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        uint64_t i, j;
        double mean[p];
        double TSS = 0.0;

        for(j=0;j<p;j++)
            mean[j] = 0.0;

        for(i=0;i<n;i++)
            for(j=0;j<p;j++)
                mean[j] += (dat[i].dim[j]/(double)n);

        for(i=0;i<n;i++)
            for(j=0;j<p;j++)
                TSS += pow(dat[i].dim[j] - mean[j], 2.0);

        return TSS;
    }
}

static double CLUSTER_computeVRC(double TSS, double SSE, uint64_t n, uint32_t k)
{
    return ((TSS * (double)(n - k)) / (SSE * (double)(k - 1)));   
}

static double CLUSTER_computeCH(double TSS, double SSE, uint64_t n, uint32_t k)
{
    double tmp = SSE / (double)(n - k);
    return ( (TSS - SSE) / (double)(k - 1)) / tmp; 
}

static void CLUSTER_initWeights(double *w, uint64_t dim)
{
    uint64_t i;
    for(i=0;i<dim;i++)
        w[i] = 1.0;
}

static void CLUSTER_computeObjectWeights(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow, eMethodType m)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        switch(m)
        {
            default:
            case METHOD_SILHOUETTE :
                {
                    CLUSTER_computeObjectWeightsViaSilhouette(dat, n, p, c, k, ow);
                }
                break;
            case METHOD_OTHER:
                {
                    WRN("Not implemented yet");
                }
                break;
        }
    }
}

static void CLUSTER_computeObjectWeightsViaSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        double a[n], b[n], s[n], sk[k], distCluster[k], dist[n][n];
        uint64_t i,j;
        uint32_t l;

        // Initilize sik
        for(l=0;l<k;l++) 
            sk[l] = 0.0;

        // Create the distance matrix of i vs j
        for(i=0;i<n;i++)
            for (j=0;j<n;j++)
                dist[i][j]= CLUSTER_computeDistancePointToPoint(&(dat[i]), &(dat[j]), p, DISTANCE_EUCLIDEAN);

        for(i=0;i<n;i++)
        {
            // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
            double d = 0.0;
            for(j=0;j<n;j++)
            {
                if(j != i && dat[j].clusterID == dat[i].clusterID)
                    d += dist[i][j];

                if((c[dat[i].clusterID].nbData - 1) == 0)
                    a[i] = 0.0;
                else
                    a[i] = d/(double)(c[dat[i].clusterID].nbData - 1);
            }

            // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
            for(l=0;l<k;l++)
                distCluster[l] = 0.0;

            for(j=0;j<n;j++)
                if(dat[j].clusterID != dat[i].clusterID)
                    distCluster[dat[j].clusterID] += (dist[i][j]/c[dat[j].clusterID].nbData);
            b[i] = 1.0e20;
            for(l=0;l<k;l++)
                if(l != dat[i].clusterID && distCluster[l] != 0 && distCluster[l] < b[i])
                    b[i] = distCluster[l];

            // Calculate s[i]
            s[i] = (b[i] != a[i]) ?  (b[i] - a[i]) / fmax(a[i], b[i]) : 0.0;

            //SAY("s[%ld] = %lf", i, s[i]);

            if(s[i] < 0 || s[i] > 0)
                s[i] = 1 - ((s[i]+1)/2); // Rescale silhouette to 0-1 
            else // si = 0
                s[i] = 0.5;

            // Calculate sum of s[i] per cluster 
            sk[dat[i].clusterID] += s[i]; 
        }

        // Calculate object weights
        for(i=0;i<n;i++)
        {
            // The sum of weights per cluster has to be equal to the number of data per cluster
            if(c[dat[i].clusterID].nbData == 1)
                ow[i] = 1.0;
            else
                ow[i] = (s[i]/sk[dat[i].clusterID])*c[dat[i].clusterID].nbData;
            //SAY("ow[%ld] = %lf", i, ow[i]);
        }
    }
}

            //SAY("ow[%ld] = %lf", i, ow[i]);
        }
    }
}

static void CLUSTER_removeNoise(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint64_t *nToRemove, uint32_t *kToRemove)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || nToRemove == NULL || kToRemove == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        /*double dst[n], distCluster[k], dist[n][n];
        uint64_t i,j, avgDataPerCluster = 0;
        uint32_t l;

        // Initilize sik
        for(l=0;l<k;l++)
        {
            avgDataPerCluster += (c[l].nbData / k);
            distCluster[l] = 0.0;
        }

        // Create the distance matrix of i vs j
        for(i=0;i<n;i++)
            for (j=0;j<n;j++)
                dist[i][j]= CLUSTER_computeDistancePointToPoint(&(dat[i]), &(dat[j]), p, DISTANCE_EUCLIDEAN);

        for(i=0;i<n;i++)
        {
            // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
            double d = 0.0;
            for(j=0;j<n;j++)
            {
                if(j != i && dat[j].clusterID == dat[i].clusterID)
                    d += dist[i][j];

                if((c[dat[i].clusterID].nbData - 1) == 0)
                    dst[i] = 0.0;
                else
                    dst[i] = d/(double)(c[dat[i].clusterID].nbData - 1);
            }

            // Update average distance point to point per cluster 
            distCluster[dat[i].clusterID] += dst[i] / (double)c[dat[i].clusterID].nbData;
        }

        for(i=0;i<n;i++)
        {
            SAY("dist[%ld] = %lf, cluster = %d, avgDistPerCluster[%d] = %lf", i, dst[i], dat[i].clusterID, dat[i].clusterID, distCluster[dat[i].clusterID]);
            if(dst[i] > (distCluster[dat[i].clusterID]*1.5))
            {
                //if(dist[i] > (avgDistPerCluster[dat[i].clusterID]*1.25))
                c[dat[i].clusterID].nbData--;
                dat[i].clusterID = k;
            }
        }

        for(l=0;l<k;l++)
        {
            SAY("c[%d].nbData = %ld, avgDataPerCluster = %ld", l, c[l].nbData, avgDataPerCluster);
            if(c[l].nbData < (avgDataPerCluster / 1.5))
            {
                WRN("Cluster %d is a potential group of noise !", l);
                for(i=0;i<n;i++)
                    if(dat[i].clusterID == l)
                    {
                        c[dat[i].clusterID].nbData--;
                        dat[i].clusterID = k; // k is the cluster of noise
                    }
            }
        }*/

        uint32_t l;
        uint64_t i;
        double avgDistPerCluster[k];
        double dist[n];
        uint64_t avgDataPerCluster = 0;

        // Initialisation
        for(l=0;l<k;l++)
        {
            avgDataPerCluster += (c[l].nbData / k);
            avgDistPerCluster[l] = 0.0;
        }

        // Calculate for each point the distance with its cluster mean
        for(i=0;i<n;i++)
        {
            dist[i] = sqrt(CLUSTER_computeSquaredDistancePointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN));
            //SAY("dist[%ld] = %lf", i, dist[i]);
            avgDistPerCluster[dat[i].clusterID] += (dist[i] / c[dat[i].clusterID].nbData); 
        }

        // Handle noise
        for(l=0;l<k;l++)
        {
            SAY("c[%d].nbData = %ld, avgDataPerCluster = %ld", l, c[l].nbData, avgDataPerCluster);
            if(c[l].nbData < (avgDataPerCluster / 1.5))
            {
                *kToRemove++;
                WRN("Cluster %d is a potential group of noise !", l);
                for(i=0;i<n;i++)
                    if(dat[i].clusterID == l)
                    {
                        c[dat[i].clusterID].nbData--;
                        *nToRemove++;
                        dat[i].clusterID = k; // k is the cluster of noise
                    }
            }
        }

        for(i=0;i<n;i++)
        {
            //SAY("dist[%ld] = %lf, cluster = %d, avgDistPerCluster[%d] = %lf", i, dist[i], dat[i].clusterID, dat[i].clusterID, avgDistPerCluster[dat[i].clusterID]);
            if(dist[i] > (avgDistPerCluster[dat[i].clusterID]*1.5))
            {
                c[dat[i].clusterID].nbData--;
                *nToRemove++;
                dat[i].clusterID = k;
            }
        }

        if((k - *kToRemove) < 2 || (n - *nToRemove) < 2)
        {
            *kToRemove = 0;
            *nToRemove = 0;
        }
    }
}

static double CLUSTER_computeSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *fw, double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double SSE = 0.0;
        uint64_t i;

        for(i=0;i<n;i++)
        {
            // Compute only real data
            if(dat[i].clusterID != k)
            {
                SSE += CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN, fw, ow[i]); 
            }
        }

        return SSE;
    }
}
