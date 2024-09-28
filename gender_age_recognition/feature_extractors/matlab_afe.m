% gender

Xgender = readmatrix('/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/spcds_gender_matlab_full.csv', ...
    delimitedTextImportOptions('NumVariables',2));
Xgender = Xgender(2:end, :);

% age

Xage2bins = readmatrix("/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/spcds_age2bins_matlab_full.csv", ...
    delimitedTextImportOptions('NumVariables',2));
Xage2bins = Xage2bins(2:end, :);

Xage4bins = readmatrix("/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/spcds_age4bins_matlab_full.csv", ...
    delimitedTextImportOptions('NumVariables',2));
Xage4bins = Xage4bins(2:end, :);

% age female, male splitted

Xage2female = readmatrix("/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/spcds_age2split_matlab_full_female.csv", ...
    delimitedTextImportOptions('NumVariables',2));
Xage2female = Xage2female(2:end, :);

Xage2male = readmatrix("/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/spcds_age2split_matlab_full_male.csv", ...
    delimitedTextImportOptions('NumVariables',2));
Xage2male = Xage2male(2:end, :);

Xage4female = readmatrix("/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/spcds_age4split_matlab_full_female.csv", ...
    delimitedTextImportOptions('NumVariables',2));
Xage4female = Xage4female(2:end, :);

Xage4male = readmatrix("/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/spcds_age4split_matlab_full_male.csv", ...
    delimitedTextImportOptions('NumVariables',2));
Xage4male = Xage4male(2:end, :);

sr = 8000;
FFTlength = 256;

afe = audioFeatureExtractor(SampleRate=sr, ...
    Window=hann(FFTlength,"periodic"), ...
    OverlapLength=(FFTlength * 0.500), ...
    melSpectrum=true, ...
    mfcc=true, ...
    mfccDelta=true, ...
    mfccDeltaDelta=true, ...
    spectralCentroid=true, ...
    spectralCrest=true, ...
    spectralDecrease=true, ...
    spectralEntropy=true, ...
    spectralFlatness=true, ...
    spectralFlux=true, ...
    spectralKurtosis=true, ...
    spectralRolloffPoint=true, ...
    spectralSkewness=true, ...
    spectralSlope=true, ...
    spectralSpread=true, ...
    pitch=true);

setExtractorParameters(afe,"melSpectrum", "NumBands", 26);

% gender

filename = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/ds_gender.mat";

ads = [];
for i = 1:length(Xgender)
    [x, sr_in] = audioread(string(Xgender(i,2)));
    if (sr ~= sr_in)
        [x, sr] = audioresample(x, sr_in, sr);
    end
    features = extract(afe,x);
    ads = [ads features];
end

save(filename, "ads")

% age

filename = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/ds_age2bins.mat"

ads = [];
for i = 1:length(Xage2bins)
    [x, sr_in] = audioread(string(Xage2bins(i,2)));
    if (sr ~= sr_in)
        [x, sr] = audioresample(x, sr_in, sr);
    end
    features = extract(afe,x);
    ads = [ads features];
end

save(filename, "ads")

filename = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/ds_age4bins.mat"

ads = [];
for i = 1:length(Xage4bins)
    [x, sr_in] = audioread(string(Xage4bins(i,2)));
    if (sr ~= sr_in)
        [x, sr] = audioresample(x, sr_in, sr);
    end
    features = extract(afe,x);
    ads = [ads features];
end

save(filename, "ads")

% age female, male splitted

filename = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/ds_age2split_female.mat"

ads = [];
for i = 1:length(Xage2female)
    [x, sr_in] = audioread(string(Xage2female(i,2)));
    if (sr ~= sr_in)
        [x, sr] = audioresample(x, sr_in, sr);
    end
    features = extract(afe,x);
    ads = [ads features];
end

save(filename, "ads")

filename = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/ds_age2split_male.mat"

ads = [];
for i = 1:length(Xage2male)
    [x, sr_in] = audioread(string(Xage2male(i,2)));
    if (sr ~= sr_in)
        [x, sr] = audioresample(x, sr_in, sr);
    end
    features = extract(afe,x);
    ads = [ads features];
end

save(filename, "ads")

filename = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/ds_age4split_female.mat"

ads = [];
for i = 1:length(Xage4female)
    [x, sr_in] = audioread(string(Xage4female(i,2)));
    if (sr ~= sr_in)
        [x, sr] = audioresample(x, sr_in, sr);
    end
    features = extract(afe,x);
    ads = [ads features];
end

save(filename, "ads")

filename = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/ds_age4split_male.mat"

ads = [];
for i = 1:length(Xage4male)
    [x, sr_in] = audioread(string(Xage4male(i,2)));
    if (sr ~= sr_in)
        [x, sr] = audioresample(x, sr_in, sr);
    end
    features = extract(afe,x);
    ads = [ads features];
end

save(filename, "ads")

% end