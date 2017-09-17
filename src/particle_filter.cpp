/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define PARTICLES_COUNT 500

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{

	normal_distribution<double> distx(x, std[0]);
	normal_distribution<double> disty(y, std[1]);
	normal_distribution<double> dist_psi(theta, std[2]);

	default_random_engine random_engine;

	for (int i = 0; i < PARTICLES_COUNT; i++)
	{
		Particle particle;
		particle.x = distx(random_engine);
		particle.id = i;
		particle.y = disty(random_engine);
		particle.theta = dist_psi(random_engine);
		particle.weight = 1;
		particles.push_back(particle);
		weights.push_back(1);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{

	default_random_engine random_engine;

	for (int i = 0; i < PARTICLES_COUNT; i++)
	{
		if (fabs(yaw_rate) < 0.0001)
		{
			particles[i].x = particles[i].x + delta_t * velocity * cos(particles[i].theta);
			particles[i].y = particles[i].y + delta_t * velocity * sin(particles[i].theta);
		}
		else
		{
			particles[i].x = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + (delta_t * yaw_rate)) - sin(particles[i].theta));
			particles[i].y = particles[i].y + (velocity / yaw_rate) * (-cos(particles[i].theta + (delta_t * yaw_rate)) + cos(particles[i].theta));
			particles[i].theta = particles[i].theta + delta_t * yaw_rate;
		}

		normal_distribution<double> distx(particles[i].x, std_pos[0]);
		normal_distribution<double> disty(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_psi(particles[i].theta, std_pos[2]);

		particles[i].x = distx(random_engine);
		particles[i].y = disty(random_engine);
		particles[i].theta = dist_psi(random_engine);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{

	for (int i = 0; i < observations.size(); i++)
	{
		double mdist = numeric_limits<double>::max();

		for (int j = 0; j < predicted.size(); j++)
		{
			double _dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (_dist < mdist)
			{
				mdist = _dist;
				observations[i].id = predicted[j].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	weights.clear();

	for (int i = 0; i < PARTICLES_COUNT; i++)
	{
		vector<LandmarkObs> landmark_observations;

		for (int j = 0; j < observations.size(); j++)
		{
			LandmarkObs landmark_observation;
			landmark_observation.x = particles[i].x + (observations[j].x * cos(particles[i].theta)) - (observations[j].y * sin(particles[i].theta));
			landmark_observation.y = particles[i].y + (observations[j].x * sin(particles[i].theta)) + (observations[j].y * cos(particles[i].theta));
			landmark_observation.id = j;
			landmark_observations.push_back(landmark_observation);
		}

		std::vector<LandmarkObs> pred_landmarks;

		for (int k = 0; k < map_landmarks.landmark_list.size(); k++)
		{
			double _dist = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
			if (_dist <= sensor_range)
			{
				LandmarkObs landmark;
				landmark.id = map_landmarks.landmark_list[k].id_i;
				landmark.x = map_landmarks.landmark_list[k].x_f;
				landmark.y = map_landmarks.landmark_list[k].y_f;
				pred_landmarks.push_back(landmark);
			}
		}

		dataAssociation(pred_landmarks, landmark_observations);

		double weight = 1.0;

		for (int j = 0; j < landmark_observations.size(); j++)
		{
			double measX = 0.0;
			double measY = 0.0;
			double muX = 0.0;
			double muY = 0.0;

			measX = landmark_observations[j].x;
			measY = landmark_observations[j].y;

			for (int k = 0; k < pred_landmarks.size(); k++)
			{
				if (pred_landmarks[k].id == landmark_observations[j].id)
				{
					muX = pred_landmarks[k].x;
					muY = pred_landmarks[k].y;
				}
			}

			double m_const = exp(-0.5 * ((pow(measX - muX, 2.0) * std_landmark[0]) + (pow(measY - muY, 2.0) * std_landmark[1])) / (sqrt(2.0 * M_PI * std_landmark[0] * std_landmark[1])));

			if (m_const > 0)
			{
				weight = weight * m_const;
			}
		}
		weights.push_back(weight);
		particles[i].weight = weight;
	}
}

void ParticleFilter::resample()
{
	std::discrete_distribution<int> dist(weights.begin(), weights.end());
	default_random_engine random_engine;
	std::vector<Particle> particle;
	for (int i = 0; i < PARTICLES_COUNT; i++)
	{
		particle.push_back(particles[dist(random_engine)]);
	}
	particles = particle;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
