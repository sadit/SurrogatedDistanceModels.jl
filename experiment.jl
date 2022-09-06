using JSON

function getfilename(path, params; prefix="", suffix="", ignorekeys=["enctime", "buildtime"])
    parts = sort!(collect(params), by=first)
    buff = IOBuffer()
    n = length(parts)
    print(buff, prefix)
    started = false
    for (i, (k, v)) in enumerate(parts)
        v isa Tuple && continue
        v isa NamedTuple && continue
        v isa AbstractDict && continue
        v isa AbstractArray && continue
        k in ignorekeys && continue
        if started
            print(buff, "--")
        else
            started = true
        end
        
        print(buff, "$k=$v")
    end
    
    print(buff, suffix)
    joinpath(path, String(take!(buff)))
end

function test_index(Gold, index, queries, k, kscale)
    GC.enable(false)
    searchtime = @elapsed I, D = searchbatch(index, queries, k * kscale)
    GC.enable(true)
    recall = Gold === nothing ? 1.0 : macrorecall(Gold, I)
    I, D, searchtime, recall, Base.summarysize(index)
end

function test_exhaustive(path, Gold, db, queries, dist, params_, k, kscalelist)
    params_["method"] = "exhaustive"
    params_["ksearch"] = k

    index = ExhaustiveSearch(; db, dist)
    params_["buildtime"] = 0.0
    
    II = nothing

    for kscale in kscalelist
        params_["kscale"] = kscale
        outbase = getfilename(path, params_)
        outdata = outbase * ".data.jld2"
        outmeta = outbase * ".meta.json"
       
        if isfile(outmeta)
            I, searchtime, recall, mem = load(outdata, "I", "searchtime", "recall", "mem")
        else
            I, D, searchtime, recall, mem = test_index(Gold, index, queries, k, kscale)
            jldsave(outdata; I, D, searchtime, recall, mem)
        end

        params = copy(params_)
        params["mem"] = mem
        params["searchtime"] = searchtime
        params["recall"] = recall

        kscale == 1 && (II = I)

        open(outmeta, "w") do f
            println(f, json(params))
        end
    end
    
    II
end

function test_searchgraph(path, Gold, db, queries, dist, params_, k, kscalelist, minrecall=0.9)
    params_["method"] = "ABS"
    params_["minrecall"] = minrecall
    params_["ksearch"] = k
    IDX = []

    function get_index()
        length(IDX) == 1 && return IDX[1]
        index = SearchGraph(; dist, db, verbose=false)
        buildtime = @elapsed index!(index)
        minrecall > 0 && optimize!(index, MinRecall(minrecall))
        push!(IDX, (index, buildtime))
        index, buildtime
    end

    for kscale in kscalelist
        params_["kscale"] = kscale
        outbase = getfilename(path, params_)
        outdata = outbase * ".data.jld2"
        outmeta = outbase * ".meta.json"
        isfile(outmeta) && continue

        index, buildtime = get_index()
        I, D, searchtime, recall, mem = test_index(Gold, index, queries, k, kscale)
        params = copy(params_)
        params["mem"] = mem
        params["searchtime"] = searchtime
        params["recall"] = recall
        params["buildtime"] = buildtime

        jldsave(outdata; I, D)
        open(outmeta, "w") do f
            println(f, json(params))
        end
    end
end
