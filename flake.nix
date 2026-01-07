{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];

    forAllSystems = nixpkgs.lib.genAttrs systems;
  in {
    devShells = forAllSystems (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };

        fhsEnv = pkgs.buildFHSEnv {
          name = "pythonFHS";

          targetPkgs = pkgs:
            with pkgs; [
              # Python tooling
              uv

              # Build toolchain for C extensions
              gcc
              gfortran
              pkg-config
              cmake

              # Runtime C/C++ libs
              stdenv.cc.cc.lib

              # NumPy native deps
              openblas
              zlib
              openssl
            ];

          profile =
            /*
            bash
            */
            ''
              if [ ! -f "pyproject.toml" ]; then
                echo "Initializing uv project..."
                uv init
              fi

              if [ ! -d ".venv" ]; then
                echo "Creating Python virtual environment with uv..."
                uv venv
              fi

              source .venv/bin/activate
            '';

          runScript = "bash";
        };
      in {
        default = fhsEnv.env;
      }
    );
  };
}
