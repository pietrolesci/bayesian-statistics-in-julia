using DataFrames, CSV
using Plots, StatsPlots
using TextAnalysis

# set working directory to the directory of this file
cd(@__DIR__)
println("Current directory: $(pwd())")

readdir("data")
path = "data/test_en.csv"

df = CSV.read(path, DataFrame)

df[:, "text_doc"] = map(StringDocument, df[:, :text])

function preprocess(t)
    prepare!(
        t,
        strip_corrupt_utf8 |
        strip_html_tags |
        strip_articles |
        strip_indefinite_articles |
        strip_definite_articles |
        strip_prepositions |
        strip_pronouns |
        strip_stopwords |
        strip_numbers |
        strip_non_letters |
        strip_sparse_terms |
        strip_frequent_terms |
        strip_punctuation |
        strip_whitespace,
    )
    # stem!(t)
    remove_case!(t)
    TextAnalysis.text(t)
end

preprocess.(df[:, "text_doc"])

map(preprocess, df[:, "text_doc"])

df[:, :label]

df[:, "Carat"]  # returns copy
df[!, "Carat"]  # returns original
df.Carat  # returns original

select(df, ["Carat"])  # returns dataframe


filter(row -> row.Carat > 1, df)
df[df[:, :Carat].>1, :]

groupby(df, :Cut) |> x -> combine(x, :Price => mean)
