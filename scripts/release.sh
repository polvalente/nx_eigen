#!/usr/bin/env bash
# Release script for nx_eigen
# Usage: ./scripts/release.sh <version>

set -e

VERSION=$1

if [ -z "$VERSION" ]; then
  echo "Usage: $0 <version>"
  echo "Example: $0 0.1.0"
  exit 1
fi

# Remove 'v' prefix if present
VERSION=${VERSION#v}

echo "🚀 Preparing release v${VERSION}"

# Update version in mix.exs
echo "📝 Updating version in mix.exs..."
sed -i.bak "s/@version \".*\"/@version \"${VERSION}\"/" mix.exs
rm mix.exs.bak

# Commit version change
echo "💾 Committing version change..."
git add mix.exs
git commit -m "Bump version to ${VERSION}"

# Create and push tag
echo "🏷️  Creating tag v${VERSION}..."
git tag "v${VERSION}"
git push origin main
git push origin "v${VERSION}"

echo ""
echo "✅ Tag pushed! GitHub Actions will now build precompiled binaries."
echo ""
echo "Next steps:"
echo "1. Wait for GitHub Actions to complete: https://github.com/YOUR_USERNAME/nx_eigen/actions"
echo "2. Run: MIX_ENV=prod mix elixir_make.checksum --all --print --ignore-unavailable"
echo "3. Commit the checksum file: git add checksum-nx_eigen.exs && git commit -m 'Add checksums for v${VERSION}'"
echo "4. Push: git push origin main"
echo "5. Publish to Hex: mix hex.publish"
