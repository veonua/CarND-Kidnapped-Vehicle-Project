/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *
 *  Modified: Nov 11, 2017
 *      Andrew Matuk
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>
#include <assert.h>

#include "particle_filter.h"

using namespace std;

static std::default_random_engine rand_engine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

    int id=0;
    std::generate_n(
            std::back_inserter(particles),
            num_particles,
            [&]() {
                return Particle{id++, dist_x(rand_engine), dist_y(rand_engine), dist_theta(rand_engine), 1.0};
            });


    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    if (fabs(yaw_rate) < 1.e-5) {
        for (auto& p : particles) {
            p.x += velocity * delta_t * cos(p.theta) + dist_x(rand_engine);
            p.y += velocity * delta_t * sin(p.theta) + dist_y(rand_engine);
        }
    } else {
        for (auto& p : particles) {
            p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta)) + dist_x(rand_engine);
            p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t)) + dist_y(rand_engine);
            p.theta += yaw_rate * delta_t + dist_theta(rand_engine); // TODO: constraint
        }
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

    for (auto& o : observations) {
        o.id = std::min_element(predicted.begin(), predicted.end(),  [=] (LandmarkObs const & p1, LandmarkObs const & p2)
        {
            return dist(o.x, o.y, p1.x, p1.y) < dist(o.x, o.y, p2.x, p2.y);
        }) -> id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, const double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    assert(particles.size() > 0);
    for (auto &particle : particles) {
        // predicted <- landmarks in the sensor_range of the current particle
        vector<LandmarkObs> predicted;
        for (auto &landmark : map_landmarks.landmark_list) {  // transform_if
            if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) < sensor_range) {
                predicted.push_back(LandmarkObs {
                        landmark.id_i,
                        landmark.x_f,
                        landmark.y_f}
                );
            }
        }

        vector<LandmarkObs> map_obs(observations.size());
        std::transform(begin(observations), end(observations), begin(map_obs),
                  [&](const LandmarkObs &obs) {
                      LandmarkObs obs_m;
                      obs_m.x = particle.x + obs.x * cos(particle.theta) - obs.y * sin(particle.theta);
                      obs_m.y = particle.y + obs.x * sin(particle.theta) + obs.y * cos(particle.theta);
                      return obs_m;
                  });

        // Set associations and weights of current particle
        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;
        if (predicted.empty()) {
            particle.weight = 0.;
        } else {
            // Assign nearest predicted landmark id to map observations
            dataAssociation(predicted, map_obs);
            particle.weight = 1.;
            for (LandmarkObs &obs : map_obs) {
                associations.push_back(obs.id);
                sense_x.push_back(obs.x);
                sense_y.push_back(obs.y);

                auto landmark = find_if(
                        begin(predicted), end(predicted),
                        [&](LandmarkObs &l_) { return l_.id == obs.id; });

                assert(landmark != predicted.end());

                auto s_x = std_landmark[0];
                auto s_y = std_landmark[1];
                double obs_w = (1. / (2 * M_PI * s_x * s_y)) *
                               exp(-(pow(landmark->x - obs.x, 2) / (2 * pow(s_x, 2)) +
                                     (pow(landmark->y - obs.y, 2) / (2 * pow(s_y, 2))))
                               );
                particle.weight *= obs_w;
            }
            SetAssociations(particle, associations, sense_x, sense_y);
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::vector<double> weights;
    for (auto const &p : particles) {
        weights.push_back(p.weight);
    }

    std::discrete_distribution<int> index_dist(weights.begin(), weights.end());

    std::vector<Particle> selectedParticles(particles.size());
    std::generate(selectedParticles.begin(), selectedParticles.end(),
             [&]() -> Particle { return particles[index_dist(rand_engine)]; });
    particles.swap(selectedParticles);
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

	particle.associations= std::move(associations);
 	particle.sense_x = std::move(sense_x);
 	particle.sense_y = std::move(sense_y);

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
