# This script installs R packages that are not available on Conda.

# Install packages from CRAN
install.packages("simr", repos="https://cloud.r-project.org/")

# Install flexplot from GitHub using the devtools package, which was already installed via conda
devtools::install_github("dustinfife/flexplot")
