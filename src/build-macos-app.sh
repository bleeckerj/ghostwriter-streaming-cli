#!/bin/bash
set -e

APP_NAME="Ghostwriter"
BINARY_NAME="ghostwriter_streaming_cli"
APP_BUNDLE="$APP_NAME.app"
MACOS_DIR="$APP_BUNDLE/Contents/MacOS"
RESOURCES_DIR="$APP_BUNDLE/Contents/Resources"

# 1. Build the Rust binary
cargo build --release

# 2. Create the .app bundle structure
rm -rf "$APP_BUNDLE"
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# 3. Copy the binary
# Move the binary to a new name
cp "target/release/$BINARY_NAME" "$MACOS_DIR/$BINARY_NAME"
chmod +x "$MACOS_DIR/$BINARY_NAME"

# Create a shell script as the main executable
cat > "$MACOS_DIR/$APP_NAME" <<EOS
#!/bin/bash
DIR="\$(cd "\$(dirname "\$0")" && pwd)"
open -a Terminal "\$DIR/$BINARY_NAME"
EOS
chmod +x "$MACOS_DIR/$APP_NAME"

# 4. Create Info.plist
cat > "$APP_BUNDLE/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>nearfuturelaboratory.ghostwriter.cli</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
EOF

echo "✅ Built $APP_BUNDLE"
echo "You can now double-click $APP_BUNDLE in Finder."