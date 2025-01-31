#!/bin/bash

## Reads a geo file to generate a mesh for SOD2D
## Usage: ./genMesh.sh <geo_file> <elements>(opt) <order>(opt)

# Pass the geo file name as an argument
if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    echo "Usage: ./genMesh.sh <geo_file> <elements>(opt) <order>(opt)"
    exit 1
fi

geo_file=$1.geo
mesh_file=${geo_file%.geo}.msh
echo "Generating mesh for $geo_file"

# 2nd argument is number of elements per side
if [ $# -eq 2 ]; then
    elements=$2
    echo "Number of elements per side: $elements"
fi

# Modify the 2nd line of the geo file
if [ $# -eq 2 ]; then
    sed -i "2s/.*/n = $elements;/" $geo_file
fi

# 3rd argument is the element order
if [ $# -eq 3 ]; then
    order=$3
    echo "Element order: $order"
fi

# Modify line 36 of the geo file
if [ $# -eq 3 ]; then
    sed -i "36s/.*/Mesh.ElementOrder = $order;/" $geo_file
fi

# Check if GMSH is installed/loaded
if ! command -v gmsh &> /dev/null
then
    echo "GMSH could not be found"
    exit 1
fi

# Generate the mesh (assumes a periodic cube)
gmsh $geo_file -o $mesh_file -0

# Check if HDF5 is installed/loaded
if ! command -v h5pcc &> /dev/null
then
    echo "HDF5 could not be found"
    exit 1
fi

# Check if Python is installed/loaded
if ! command -v python &> /dev/null
then
    echo "Python could not be found"
    exit 1
fi

# Convert the mesh to HDF5 format
python gmsh2sod2d.py cube -p 1 -r $order