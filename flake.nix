{
  description = "Dev shell for C/C++";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ...}: let
    pkgs = nixpkgs.legacyPackages."x86_64-linux";
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
		
	  packages = with pkgs; [
	    cmake
	    libgcc
	    llvmPackages_19.libcxx
	    libpkgconf
	  ];
	
	  shellHook = ''
	    export CXX=g++
	    export CC=gcc
	    echo "Development environment for C/C++ is ready!"
	  '';
    };
  };
}
