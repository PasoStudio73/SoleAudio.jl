# ---------------------------------------------------------------------------- #
#                              abstract structures                             #
# ---------------------------------------------------------------------------- #
abstract type AbstractMLDataFrame end
abstract type AbstractMLMachine end

# ---------------------------------------------------------------------------- #
#                                     utils                                    #
# ---------------------------------------------------------------------------- #
check_dataframe_type(df::AbstractDataFrame) = all(col -> eltype(col) <: AbstractVector{<:AbstractFloat}, eachcol(df))

# ---------------------------------------------------------------------------- #
#                                data structures                               #
# ---------------------------------------------------------------------------- #
"""
MLD struct is a container for machine learning datasets that extends AbstractMLDataFrame.

Type Parameters:
D: holds the DataFrame type
C: stores the categorical array type for class labels
V: represents the string type for variable names

Fields:
df: stores the feature data as a DataFrame
cls: contains the class labels as a categorical array
vnames: holds feature names as strings

Constructors:
Primary constructor validates data consistency
Convenience constructor auto-generates variable names
Type checking ensures DataFrame contains only float vectors

Interface:
Implements DataFrame-like indexing
Returns tuples of (data, labels, varnames)
"""
struct MLD{D<:AbstractDataFrame,C<:AbstractCategoricalArray,V<:AbstractString} <: AbstractMLDataFrame
    df::D
    cls::C
    vnames::AbstractVector{V}

    MLD(df::D, cls::C) where {D<:AbstractDataFrame,C<:AbstractCategoricalArray} = MLD(df, cls, names(df))
    function MLD(df::D, cls::C, vnames::AbstractVector{V}=names(df)) where {D<:AbstractDataFrame,C<:AbstractCategoricalArray,V<:AbstractString}
        check_dataframe_type(df)      || throw(ArgumentError("DataFrame must contain only Vector{<:AbstractFloat} columns"))
        size(df, 1) == length(cls)    || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))
        size(df, 2) == length(vnames) || throw(ArgumentError("Number of columns in DataFrame must match length of variable names"))
        new{D,C,V}(df, cls, vnames)
    end
end
function Base.show(io::IO, ::MIME"text/plain", x::MLD)
    println(io, "MLD dataset with $(size(x.df, 1)) samples and $(size(x.df, 2)) features")
    println(io, "\nDataFrame:")
    show(io, x.df)
    println(io, "\n\nClass labels:")
    show(io, x.cls)
end
Base.getindex(x::MLD, i, j) = (x.df[i,j], x.cls[i], x.vnames[j])













