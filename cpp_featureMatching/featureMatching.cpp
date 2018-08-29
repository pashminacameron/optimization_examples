// needs c++11
#include <chrono>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>

#define FEATURE_SIZE 128
#define NUM_FEATURES 100000
#define CODE_BLOAT 2048 // vary from 8 to 2048 in powers of 2
#define DATA_TYPE float // or int

namespace cn = std::chrono;

cn::time_point<cn::high_resolution_clock> startTimer()
{
	return cn::high_resolution_clock::now();
}

double endTimer(cn::time_point<cn::high_resolution_clock> t1)
{
	auto t2 = cn::high_resolution_clock::now();
	cn::duration<double> diff = t2 - t1;
	return diff.count() * 1000;
}

//Aggregate struct 
struct PointFeature 
{
    DATA_TYPE x;
    DATA_TYPE y;
	DATA_TYPE otherPointData[CODE_BLOAT];
    DATA_TYPE feature[FEATURE_SIZE];
};

//Parallel vectors
struct Point
{
    DATA_TYPE x;
    DATA_TYPE y;
	DATA_TYPE otherPointData[CODE_BLOAT];
};

struct Feature
{
    DATA_TYPE feature[FEATURE_SIZE];
};

void genRandomFeature(DATA_TYPE * feature)
{
	srand(111970);
	for (int j = 0; j < FEATURE_SIZE; j++)
        feature[j] = (DATA_TYPE)rand() / RAND_MAX;
	
	return;
}

void computeDistancesAggregateVector(const DATA_TYPE * testFeature, const std::vector<PointFeature> & ptFeatures, std::vector<float> & distances)
{
    for(int i = 0; i < ptFeatures.size(); i++)
    {
        const DATA_TYPE * feature = &ptFeatures[i].feature[0];
        DATA_TYPE sum = 0;
		for (int j = 0; j < FEATURE_SIZE; j++)
			sum += (testFeature[j] - feature[j]) * (testFeature[j] - feature[j]);
        distances[i] = std::sqrt(sum);
    }
}

void computeDistancesParallelVector(const DATA_TYPE * testFeature, const std::vector<Feature> & features, std::vector<float> & distances)
{
    for(int i = 0; i < features.size(); i++)
    {
        const DATA_TYPE * feature = &features[i].feature[0];
        DATA_TYPE sum = 0;
		for (int j = 0; j < FEATURE_SIZE; j++)
			sum += (testFeature[j] - feature[j]) * (testFeature[j] - feature[j]);
        distances[i] = std::sqrt(sum);
    }
}

int main(int argc, const char * argv[])
{
	// Generate some random features to populate aggregate point features
	std::vector<PointFeature> pointFeatures;
    pointFeatures.reserve(NUM_FEATURES);
    for(int i = 0; i < NUM_FEATURES; i++)
    {
        PointFeature pt;
        pt.x = i*0.1;
        pt.y = (NUM_FEATURES-i)*0.1;
        genRandomFeature(&pt.feature[0]);
        pointFeatures.emplace_back(pt);
    }
    
	// Generate some random features to populate parallel vectors of points and features
    std::vector<Point> pts;
    std::vector<Feature> features;
    pts.reserve(NUM_FEATURES);
    features.reserve(NUM_FEATURES);
    for(int i = 0; i < NUM_FEATURES; i++)
    {
        Point pt;
        pt.x = i*0.1;
        pt.y = (NUM_FEATURES -i)*0.1;
        Feature feat;
        genRandomFeature(&feat.feature[0]);
        pts.emplace_back(pt);
        features.emplace_back(feat);
    }

	// For numRuns, compute distance of each of the NUM_FEATURES features aganst the first feature
	// Add the distance to a random DATA_TYPE to stop the compiler from optimizing away the loop
	int numRuns = 100;
    std::vector<float> distances1(NUM_FEATURES,0);
    float a = 1.f;
    auto t1 = startTimer();
    for(int i = 0 ; i < numRuns; i++)
    {
        computeDistancesAggregateVector(&pointFeatures[0].feature[0], pointFeatures, distances1);
        a = a + distances1[0];
    }
    double time1 = endTimer(t1) / numRuns;
    
    std::vector<float> distances2(NUM_FEATURES,0);
    float b = 1.f;
    auto t2 = startTimer();
    for(int i = 0 ; i < numRuns; i++)
    {
        computeDistancesParallelVector(&features[0].feature[0], features, distances2);
        b = b + distances2[0];
    }
    double time2 = endTimer(t2) / numRuns;

    std::cout << CODE_BLOAT << "\t" << time1 << "\t" << time2 << std::endl;
	
	return 0;
}
