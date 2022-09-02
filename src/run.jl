using JSON

const DATAPATH = "../data/"

function getfilename(path, params; prefix="", suffix="", ignorekeys=["enctime"])
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
    joinpath(DATAPATH, String(take!(buff)))
end

function test_exhaustive(Gold, db, queries, dist, params, k)
    params["method"] = "exhaustive"
    params["ksearch"] = k
    outbase = getfilename(DATAPATH, params)
    outdata = outbase * ".data.jld2"
    outmeta = outbase * ".meta.json"
    
    isfile(outmeta) && return load(outdata, "I")

    index = ExhaustiveSearch(; db, dist)
    params["buildtime"] = 0.0
    GC.enable(false)
    params["searchtime"] = @elapsed I, D = searchbatch(index, queries, k)
    GC.enable(true)
    jldsave(outdata; I, D)
    params["mem"] = Base.summarysize(index)
    params["recall"] = Gold === nothing ? 1.0 : macrorecall(Gold, I)
    open(outmeta, "w") do f
        println(f, json(params))
    end
    
    I
end

function test_searchgraph(Gold, db, queries, dist, params, k, minrecall=0.9)
    params["method"] = "ABS"
    params["minrecall"] = minrecall
    params["ksearch"] = k
    outbase = getfilename(DATAPATH, params)
    outdata = outbase * ".data.jld2"
    outmeta = outbase * ".meta.json"
    
    isfile(outmeta) && return
    
    index = SearchGraph(; dist, db, verbose=false)
    params["buildtime"] = @elapsed index!(index)
    minrecall > 0 && optimize!(index, MinRecall(minrecall))
    GC.enable(false)
    params["searchtime"] = @elapsed I, D = searchbatch(index, queries, k)
    GC.enable(true)
    jldsave(outdata; I, D)
    params["mem"] = Base.summarysize(index)
    params["recall"] = macrorecall(Gold, I)
    open(outmeta, "w") do f
        println(f, json(params))
    end
end
