function out = vilds_import_results(infile,varargin)
%% function out = vilds_import_results(infile,varargin)
%%
%% imports results from vlgp
%%
%% inputs: 
%%
%%   infile: filename with stored data
%%
%%   optional: which variables to load


    in_info = h5info(infile);

    if exist('varargin','var')
        toLoad = varargin;
    else
        toLoad = {in_info.Datasets.name};
    end
    for nn = 1:numel(toLoad)
        d = toLoad{nn};
        tmp = h5read(infile, sprintf('/%s', d));
        out.(d) = tmp;
    end
