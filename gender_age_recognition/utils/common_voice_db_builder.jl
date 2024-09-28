using CSV
using DataFrames
using Audio911
# using ConfigEnv
# dotenv()

#---------------------------------------------------------------------------------------------------#
#                                   datasets and global variables                                   #
#---------------------------------------------------------------------------------------------------#
source_db = [
    # 1-absolute path to database, 2-database filename, 3-additional path to files
    # ("/media/paso/Paso Work HD/Aclai/Mozilla_common_voice/cv-corpus-16.1-2023-12-06/de/", "import_ready.csv", "clips/"),
    # ("/media/paso/Paso Work HD/Aclai/Mozilla_common_voice/cv-corpus-16.1-2023-12-06/en/", "import_ready_pt1.csv", "clips/"),
    # ("/media/paso/Paso Work HD/Aclai/Mozilla_common_voice/cv-corpus-16.1-2023-12-06/en/", "import_ready_pt2.csv", "clips/"),
    # ("/media/paso/Paso Work HD/Aclai/Mozilla_common_voice/cv-corpus-16.1-2023-12-06/es/", "import_ready.csv", "clips/"),
    # ("/media/paso/Paso Work HD/Aclai/Mozilla_common_voice/cv-corpus-16.1-2023-12-06/fr/", "import_ready.csv", "clips/"),
    # ("/media/paso/Paso Work HD/Aclai/Mozilla_common_voice/cv-corpus-16.1-2023-12-06/it/", "import_ready.csv", "clips/"),
    # ("/media/paso/Paso Work HD/Aclai/Mozilla_common_voice/cv-corpus-16.1-2023-12-06/ja/", "import_ready.csv", "clips/"),
    # ("/media/paso/Paso Work HD/Aclai/Mozilla_common_voice/cv-corpus-16.1-2023-12-06/sv-SE/", "import_ready.csv", "clips/"),
]
# source_db = ENV["source_db"]

dest_db = ("/home/paso/Documents/Aclai/Datasets/Common_voice_ds/", "common_voice_ds.csv")
# dest_db = ENV["dest_db"]

# audiofile path must be in first position!
labels_db = ["path", "client_id", "gender", "age"]
# increasing order
lengths_db = [3, 6]

sr = 8000 # database sample rate
trim_threshold = 25

function voice_db_builder(
    source_db::Vector{<:Tuple{Vararg{String}}},
    dest_db::NTuple{2,String},
    labels_db::Vector{String},
    lengths_db::Vector{Int64},
    sr_db::Int64
)
    #-----------------------------------------------------------------------------------------------#
    #                             result dir and existing .csv check                                #
    #-----------------------------------------------------------------------------------------------#
    @info "Starting..."

    dest_path, dest_file = dest_db
    spl_nm = split(dest_file, ".")

    if !isdir(dest_path)
        mkdir(dest_path)
        cd(dest_path)
        for i in lengths_db
            mkdir("$i/")
            mkdir("$i/Wavfiles/")

            open(string(dest_path, "$i/", spl_nm[1], "_$i.", spl_nm[2]), "w") do f
                CSV.write(f, [], writeheader=true, header=labels_db)
            end
        end
    end

    #-----------------------------------------------------------------------------------------------#
    #                                        load datasets                                          #
    #-----------------------------------------------------------------------------------------------#

    for i in source_db
        source_path, source_file, wav_path = i
        @info "dataset: $source_path$source_file"
        df = DataFrame(CSV.File(string(source_path, source_file)))

        for j in eachrow(df)
            x, sr = load_audio(string(source_path, wav_path, j.path), sr=sr_db)

            #-----------------------------------------------------------------------------------------------#
            #                                validate sample legth and save                                 #
            #-----------------------------------------------------------------------------------------------#
            for k in lengths_db
                # check initial length of sample
                if length(x) >= k * sr
                    x = normalize_audio(x)
                    x = speech_detector(x, sr)
                    # check length of sample after speech_detector
                    if length(x) >= k * sr
                        y = x[1:k*sr]
                        j.path[end-2:end] == "mp3" && (j.path = replace(j.path, ".mp3" => ".wav"))
                        save_audio(string(dest_path, "$k/Wavfiles/", j.path), y, sr=sr_db)
                        ## append in csv files
                        CSV.write(
                            string(dest_path, "$k/", spl_nm[1], "_$k.", spl_nm[2]),
                            DataFrame(j),
                            writeheader=false,
                            append=true
                        )
                    end
                end
            end
        end
    end
    @info "Process terminated."
end

voice_db_builder(source_db, dest_db, labels_db, lengths_db, sr)
