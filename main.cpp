#include <nlohmann/json.hpp>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <set>
#include <iomanip>
#include <numeric>
#include <cmath>
using namespace std;
using json = nlohmann::json;

using Chromosome = vector<int>;
using Population = vector<Chromosome>;

int N_CAPTEURS = 0;
int N_EMPLACEMENTS = 0;
int TAILLE_POPULATION = 50;
int TAILLE_CHROMOSOME = N_EMPLACEMENTS * N_CAPTEURS;
int MAX_GENERATIONS = 100;

vector<pair<double, double>> emplacements;
vector<pair<double, double>> points_interet;
vector<vector<double>> matrice_distance;
vector<double> rayons_capteurs;

void chargerDonneesDepuisBlocJSON(const json& bloc) {
    emplacements.clear();
    points_interet.clear();
    rayons_capteurs.clear();
    matrice_distance.clear();

    for (const auto& e : bloc["locations"])
        emplacements.emplace_back(e["x"], e["y"]);

    for (const auto& p : bloc["pois"])
        points_interet.emplace_back(p["x"], p["y"]);

    for (const auto& capteur : bloc["sensors"])
        rayons_capteurs.push_back(capteur["range"]);

    N_CAPTEURS = rayons_capteurs.size();
    N_EMPLACEMENTS = emplacements.size();
    TAILLE_CHROMOSOME = N_CAPTEURS * N_EMPLACEMENTS;
}

double calculerDistance(const pair<double, double>& a, const pair<double, double>& b) {
    return sqrt(pow(a.first - b.first, 2) + pow(a.second - b.second, 2));
}

void calculerMatriceDistance() {
    matrice_distance.resize(N_EMPLACEMENTS, vector<double>(points_interet.size()));
    for (int i = 0; i < N_EMPLACEMENTS; ++i)
        for (int j = 0; j < points_interet.size(); ++j)
            matrice_distance[i][j] = calculerDistance(emplacements[i], points_interet[j]);
}

double fonctionFitness(const Chromosome& individu) {
    vector<bool> couverts(points_interet.size(), false);
    int capteurs_utiles = 0;

    for (int i = 0; i < N_CAPTEURS; ++i) {
        if (individu[i] == 0) continue;

        int e = individu[i] - i * N_EMPLACEMENTS - 1;
        if (e < 0 || e >= matrice_distance.size()) continue;

        double rayon = rayons_capteurs[i];
        bool a_couvert = false;

        for (int j = 0; j < points_interet.size(); ++j) {
            if (!couverts[j] && matrice_distance[e][j] <= rayon) {
                couverts[j] = true;
                a_couvert = true;
            }
        }

        if (a_couvert) capteurs_utiles++;
    }

    int nb_couverts = count(couverts.begin(), couverts.end(), true);

    if (nb_couverts == points_interet.size()) {
        // Objectif : minimiser capteurs_utiles
        return 10000.0 - capteurs_utiles; // grande valeur - nombre de capteurs
    } else {
        // Pénaliser fort si la solution n'est pas valide
        return -1000.0 + nb_couverts;
    }
}


Chromosome tournoiSelection(const Population& population, int k = 5) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, population.size() - 1);
    Chromosome meilleur_individu;
    double meilleure_fitness = -1e9;

    for (int i = 0; i < k; ++i) {
        int index = distrib(gen);
        const Chromosome& candidat = population[index];
        double f = fonctionFitness(candidat);
        if (f > meilleure_fitness) {
            meilleure_fitness = f;
            meilleur_individu = candidat;
        }
    }
    return meilleur_individu;
}

pair<Chromosome, Chromosome> croisement(const Chromosome& parent1, const Chromosome& parent2) {
    int taille = parent1.size();
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(1, taille - 2);
    int point = dist(gen);

    Chromosome enfant1 = parent1;
    Chromosome enfant2 = parent2;
    for (int i = point; i < taille; ++i)
        swap(enfant1[i], enfant2[i]);

    return {enfant1, enfant2};
} 
// تابع الكود الحالي وستجد فيه...
Chromosome greedyInitialisation() {
    Chromosome individu(N_CAPTEURS, 0);
    vector<bool> couverts(points_interet.size(), false);

    for (int i = 0; i < N_CAPTEURS; ++i) {
        int best_emplacement = -1;
        int best_cover = 0;

        for (int e = 0; e < N_EMPLACEMENTS; ++e) {
            int cover_count = 0;
            for (int j = 0; j < points_interet.size(); ++j)
                if (!couverts[j] && matrice_distance[e][j] <= rayons_capteurs[i])
                    cover_count++;

            if (cover_count > best_cover) {
                best_cover = cover_count;
                best_emplacement = e;
            }
        }

        if (best_emplacement != -1) {
            individu[i] = best_emplacement + i * N_EMPLACEMENTS + 1;

            for (int j = 0; j < points_interet.size(); ++j)
                if (matrice_distance[best_emplacement][j] <= rayons_capteurs[i])
                    couverts[j] = true;
        }
    }

    return individu;
}


void mutation(Chromosome& individu, double taux_mutation = 0.1) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> proba(0.0, 1.0);

    for (int i = 0; i < individu.size(); ++i) {
        if (proba(gen) < taux_mutation) {
            int base = i * N_EMPLACEMENTS;
            uniform_int_distribution<> dist_emplacement(base + 1, base + N_EMPLACEMENTS);
            individu[i] = dist_emplacement(gen);
        }
    }
}

Population initialiserPopulation() {
    Population population;
    // أول فرد: باستخدام greedy
    population.push_back(greedyInitialisation());

    random_device rd;
    mt19937 gen(rd());

    for (int i = 1; i < TAILLE_POPULATION; ++i) {
        Chromosome individu;
        for (int j = 0; j < N_CAPTEURS; ++j) {
            uniform_real_distribution<> chance(0.0, 1.0);
            if (chance(gen) < 0.3)
                individu.push_back(0);
            else {
                int base = j * N_EMPLACEMENTS;
                uniform_int_distribution<> distrib(base + 1, base + N_EMPLACEMENTS);
                individu.push_back(distrib(gen));
            }
        }
        population.push_back(individu);
    }

    return population;
}
void enregistrerResultat(const string& nom, int nb_poi, int nb_caps, bool faisable,
                         int utilises, int nb_poi_couverts, long long temps, int gen,
                         const string& fichier_csv) {
    ofstream fichier(fichier_csv, ios::app);
    fichier << nom << "," << nb_poi << "," << nb_caps << "," << (faisable ? "Oui" : "Non") << ","
            << utilises << "," << nb_poi_couverts << "," << temps << "," << gen << "\n";
    fichier.close();
}


int main() {
    const string fichier_json = "data1.json";
    const string fichier_csv = "resultats.csv";

    ofstream fichier_out(fichier_csv);
fichier_out << "Fichier,POI,Capteurs,Faisable,Capteurs_utilisés,POI_Couverts,Temps_ms,Generations\n";

    fichier_out.close();

    ifstream f(fichier_json);
    if (!f) {
        cerr << "❌ Impossible d'ouvrir " << fichier_json << endl;
        return 1;
    }

    json data;
    f >> data;

    for (auto& [image, bloc] : data.items()) {
        cout << "🔄 Traitement de : " << image << endl;

        auto debut = chrono::high_resolution_clock::now();

        chargerDonneesDepuisBlocJSON(bloc);
        calculerMatriceDistance();
        Population population = initialiserPopulation();

        Chromosome meilleur_global;
        double meilleur_fitness = -1e9;
        int generation_finale = -1;
        bool faisable = false;

        for (int gen = 1; gen <= MAX_GENERATIONS; ++gen) {
            double meilleure_fitness = -1e9;
            Chromosome meilleur_local;

            for (const Chromosome& ind : population) {
                vector<bool> couverts(points_interet.size(), false);
                for (int k = 0; k < N_CAPTEURS; ++k) {
                    if (ind[k] == 0) continue;
                    int e = ind[k] - k * N_EMPLACEMENTS - 1;
                    if (e < 0 || e >= matrice_distance.size()) continue;
                    double rayon = rayons_capteurs[k];
                    for (int j = 0; j < points_interet.size(); ++j)
                        if (!couverts[j] && matrice_distance[e][j] <= rayon)
                            couverts[j] = true;
                }

                bool ok = count(couverts.begin(), couverts.end(), true) == points_interet.size();
                double fit = fonctionFitness(ind);
                if (ok && fit > meilleure_fitness) {
                    meilleure_fitness = fit;
                    meilleur_local = ind;
                    generation_finale = gen;
                    faisable = true;
                }
            }

            if (faisable) {
                meilleur_global = meilleur_local;
                break;
            }

            Population nouvelle;
            for (int i = 0; i < TAILLE_POPULATION; ++i)
                nouvelle.push_back(tournoiSelection(population));
            Population croisee;
            for (int i = 0; i < TAILLE_POPULATION; i += 2) {
                auto [e1, e2] = croisement(nouvelle[i], nouvelle[(i + 1) % TAILLE_POPULATION]);
                croisee.push_back(e1);
                croisee.push_back(e2);
            }
            for (Chromosome& ind : croisee)
                mutation(ind);
            population = croisee;
        }

        auto fin = chrono::high_resolution_clock::now();
        auto temps_ms = chrono::duration_cast<chrono::milliseconds>(fin - debut).count();

       int capteurs_utiles = 0;
vector<bool> couverts(points_interet.size(), false);
int nb_couverts = 0;

if (faisable) {
    for (int i = 0; i < N_CAPTEURS; ++i) {
        if (meilleur_global[i] == 0) continue;
        capteurs_utiles++;
        int e = meilleur_global[i] - i * N_EMPLACEMENTS - 1;
        double rayon = rayons_capteurs[i];

        for (int j = 0; j < points_interet.size(); ++j) {
            if (!couverts[j] && matrice_distance[e][j] <= rayon)
                couverts[j] = true;
        }
    }

    nb_couverts = count(couverts.begin(), couverts.end(), true);
}


        enregistrerResultat(image, points_interet.size(), N_CAPTEURS,
                    faisable, capteurs_utiles, nb_couverts, temps_ms,
                    generation_finale == -1 ? MAX_GENERATIONS : generation_finale,
                    fichier_csv);

    }

    cout << "\n✅ Toutes les expériences de `data1.json` ont été traitées.\n";
    return 0;
}
