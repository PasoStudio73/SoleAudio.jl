using CSV
using DataFrames

walkpath = "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/results"

cd(walkpath)

for (root, _, files) in walkdir(".")
    for file in files
        # verifica la presenza di un file spcds.out
        # verifica se è già presente un file feature_count.csv
        if startswith(file, "emovo") && endswith(file, ".out") && !isfile(string(root, "/features_count.csv"))
            open(string(root, "/features_count.csv"), "w") do f
                CSV.write(f, [], writeheader=true, header=["feature", "count"])
            end

            open(joinpath(root, file)) do f
                feat = 1
                println(joinpath(root, file))
                cont = true
                
                while cont
                    c = 0
                    for i in eachline(f)
                        if occursin("[V$feat]", i)
                            c += 1
                        end
                    end

                    if c == 0
                        cont = false
                    else
                        CSV.write(
                            string(root, "/features_count.csv"),
                            DataFrame(feature=feat, count=c),
                            writeheader=false,
                            append=true
                        )
                        feat += 1
                        seek(f, 0)
                    end
                end
            end
        end
    end
end

