import os
import warnings
import neat
import pickle
from func import load_labeled_df_from_csv, load_df_from_csv, labeling_data
warnings.filterwarnings('ignore')


def eval_genomes(genomes, config):
    nets = []
    ge = []
    # Creating individual data for every object in this generation
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 4
        nets.append(net)
        ge.append(genome)

    for i, row in labeled_data.iterrows():
        for j in range(0, len(nets)):
            net_input = (row["upper_wick_group"], row["ibs_vol_group"], row["rsi_area"], row["higher_high_lower_vol"],
                         row["Volume_higher_avg"], row["Volume_vs_prev_Vol"], row["Volume_avg_group"],
                         row["close_price_group"], row["open_price_group"], row["High_position"], row["BB_rejection"])
            output = nets[j].activate(net_input)
            ge[j].fitness -= (output[0] - row["is_max"]) ** 2


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # creating reporter that will print crucial data in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 20 generations.
    winner = p.run(eval_genomes, 20)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    # save the best genome to a file
    best_genome_path = os.path.join(local_dir, 'neat.brain')
    with open(best_genome_path, 'wb') as fp:
        pickle.dump(winner, fp)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    csv_file = str(local_dir) + '/labeled_data.csv'
    labeled_data = load_labeled_df_from_csv(csv_file)
    if labeled_data is None:
        df = load_df_from_csv()
        df = df[(df.index > '2022-01-01 00:00:00') & (df.index < '2024-01-01 00:00:00')]
        labeled_data = labeling_data(df)
        labeled_data.to_csv(csv_file)
    config_path = os.path.join(local_dir, 'neat-config')
    run(config_path)
