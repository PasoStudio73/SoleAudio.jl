using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CSV
using DataFrames, JLD2

include("../afe.jl")
include("../utils.jl")
include("speaker_recognition.jl")

#--------------------------------------------------------------------------------------#
#                                      parameters                                      #
#--------------------------------------------------------------------------------------#
jld2_path = "/home/paso/Documents/Aclai/ItaData2024/jld2_files/speaker_recognition/"

train_prefix = "srTrain"
test_prefix = "srTest"

file_regex = r"(^[a-zA-Z]*)_([0-9]*)_([0-9]*)"

# labels csv
labels_db = ["seed", "n_samples", "accuracy"]

#--------------------------------------------------------------------------------------#
#                                   load jld2 files                                    #
#--------------------------------------------------------------------------------------#
cd(jld2_path)

csv_file = string("/home/paso/Documents/Aclai/experiments_results/speaker_recognition/spkr_accuracy.csv")

open(csv_file, "w") do f
	CSV.write(f, [], writeheader = true, header = labels_db)
end

# carico in ciclo tutti i file jld2 corrispondenti a train_set e relativo test_set
# il nome del file è così composto:
# prefisso(srTrain = train set, srTest = test set)_seed_n_samples.jld2
for (root, _, files) in walkdir(".")
	for file in files
		file_info = match(file_regex, file)
		if file_info[1] == train_prefix
			@info string("loading train set, seed: ", file_info[2], ", n_samples: ", file_info[3])
			train_file = file
			test_file = string(test_prefix, "_", file_info[2], "_", file_info[3], ".jld2")
			seed = parse(Int, file_info[2])
			n_samples = parse(Int, file_info[3])

			# load datasets
			# il dataframe train_df sarà composto da 2 colonne:
			# per ogni id avrò un vettore di centroidi e uno si deviazione standard, lungo quanto il numero di audio features
			# il dataframe test_df avrà un fettore di features e un booleano che indica se il sample è di un id conosciuto
			# l'id conosciuto non verrà utilizzato, se non per creare la matrice di confusione

			train_df, train_y = load_jld2(train_file)
			test_df, test_y = load_jld2(test_file)

			test_set = build_test_set(train_df, train_y, test_df, test_y)

			check_sigma(test_set)
			check_overlap(test_set)
			check_distances(test_set)
			accuracy = calc_accuracy(test_set, train_y)

			# append in csv files
			CSV.write(
				csv_file,
				DataFrame(
					seed = seed,
					n_samples = n_samples,
					accuracy = accuracy,
				), writeheader = false,
				append = true,
			)

			# salvataggio della matrice di confusione
		end
	end
end