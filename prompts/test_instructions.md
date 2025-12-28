## Build
Go to Ruzino/build and run:
```
ninja.exe
```

But if there are only shader changes, rebuilding is not needed.

## Test Instructions
Run the following command in Ruzino/Binaries/Release:
```
.\.\headless_render.exe -u ..\..\Assets\stage.usdc -j ..\..\Assets\render_nodes_save.json -o profile_sponza.png -w 1920 -h 1080 -s 64
```

Or
```
.\Ruzino.exe somestage.usdc
```

If it is some test, just use the cpp name plus _test.exe, e.g.,
```
.\geom_algorithms_test.exe
```