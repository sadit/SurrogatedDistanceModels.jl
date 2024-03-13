using Documenter, SurrogatedDistanceModels

makedocs(;
    modules=[SurrogatedDistanceModels],
    authors="Eric S. Tellez",
    repo="https://github.com/sadit/SurrogatedDistanceModels.jl/blob/{commit}{path}#L{line}",
    sitename="SurrogatedDistanceModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://sadit.github.io/SurrogatedDistanceModels.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ],
    warnonly=true
)

deploydocs(;
    repo="github.com/sadit/SurrogatedDistanceModels.jl",
    devbranch=nothing,
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"]
)
