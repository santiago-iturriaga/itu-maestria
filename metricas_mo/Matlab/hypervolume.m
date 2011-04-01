function hypervolume(FP_file, FP_n_files)
    [hyper_fp, hyper_inst] = hypervolume_testset(FP_file, FP_n_files);
    disp(hyper_fp);
    disp(hyper_inst);

    best = max(hyper_inst);
    average = mean(hyper_inst);
    stdev = std(hyper_inst);

    disp('Best / Average / StDev');
    disp(best);
    disp(average);
    disp(stdev);

    result = zeros(1,3);
    result(1) = best * 100 / hyper_fp;
    result(2) = average * 100 / hyper_fp;
    result(3) = stdev * 100 / hyper_fp;

    disp('Best (%) / Average (%) / StDev (%)');
    disp(result);   
end

function r = find_r(instances)
    dim_instances = size(instances);
    cant_instances = dim_instances(1);

    r = [0, 0];

    for k=1:cant_instances 
        FP = load(char(instances(k)));
        FP_dim = size(FP);
        FP_dim1 = FP_dim(1); % cantidad de instancias
        FP_dim2 = FP_dim(2); % 2 objetivos

        Max_Makespan = max(FP(:,1));
        Max_WRR = max(FP(:,2));

        makespan_aux = [r(1), Max_Makespan];
        wrr_aux = [r(2), Max_WRR];

        r = [max(makespan_aux),max(wrr_aux)]
    end
end

function [v_fp, v_instances] = hypervolume_testset(fp, instances)
    dim_instances = size(instances);
    cant_instances = dim_instances(1);

    r = find_r(instances);
    disp('Ref R:');
    disp(r);

    v_fp = hypervolume_file(fp, r);
    disp('Ref Hyper:');
    disp(v_fp);

    v_instances = zeros(cant_instances,1);
    for k=1:cant_instances       
        v_instances(k) = hypervolume_file(char(instances(k)),r);
        disp('Hyper:');
        disp(v_instances(k));
    end

    disp(v_instances)
end

function v = hypervolume_file(file, FP_r)
    disp('file:');
    disp(file);

    FP = load(file);
    FP_dim = size(FP);
    FP_dim1 = FP_dim(1); % cantidad de instancias
    FP_dim2 = FP_dim(2); % 2 objetivos

    disp('----------------');
    disp('FP:');
    disp(FP);
    disp('R point:');
    disp(FP_r);

    v = hypervolume_func(FP, FP_r, 100000);
end

function v = hypervolume_func(P,r,N)
    % HYPERVOUME    Hypervolume indicator as a measure of Pareto front estimate.
    %   V = HYPERVOLUME(P,R,N) returns an estimation of the hypervoulme (in
    %   percentage) dominated by the approximated Pareto front set P (n by d)
    %   and bounded by the reference point R (1 by d). The estimation is doen
    %   through N (default is 1000) uniformly distributed random points within
    %   the bounded hyper-cuboid.  
    %
    %   V = HYPERVOLUMN(P,R,C) uses the test points specified in C (N by d).
    %
    % See also: paretofront, paretoGroup

    % Version 1.0 by Yi Cao at Cranfield University on 20 April 2008

    % Example
    %{
    % an random exmaple
    F=(randn(100,3)+5).^2;
    % upper bound of the data set
    r=max(F);
    % Approximation of Pareto set
    P=paretofront(F);
    % Hypervolume
    v=hypervolume(F(P,:),r,100000);
    %}

    % Check input and output
    error(nargchk(2,3,nargin));
    error(nargoutchk(0,1,nargout));

    P=P*diag(1./r);
    [n,d]=size(P);
    if nargin<3
        N=1000;
    end
    if ~isscalar(N)
        C=N;
        N=size(C,1);
    else
        C=rand(N,d);
    end

    fDominated=false(N,1);
    lB=min(P);
    %fcheck=all(bsxfun(@gt, C, lB),2);
    fcheck=all(C>lB(ones(N,1),:),2);

    for k=1:n
        if any(fcheck)
            %f=all(bsxfun(@gt, C(fcheck,:), P(k,:)),2);
            f=all(C(fcheck,:)>P(k(ones(sum(fcheck),1)),:),2);
            fDominated(fcheck)=f;
            fcheck(fcheck)=~f;
        end
    end

    v=sum(fDominated)/N;
end
