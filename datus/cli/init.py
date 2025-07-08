#!/usr/bin/env python3
"""
Datus Agent Initialization Command

This module provides the initialization functionality for setting up
the ~/.datus directory structure and copying necessary files.
"""

import shutil
import sys
from pathlib import Path

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DatusInitializer:
    """Class responsible for initializing Datus Agent user environment"""

    def __init__(self):
        self.user_home = Path.home()
        self.datus_dir = self.user_home / ".datus"
        self.data_dir = self.datus_dir / "data"
        self.conf_dir = self.datus_dir / "conf"
        self.template_dir = self.datus_dir / "template"
        self.sample_dir = self.datus_dir / "sample"

    def create_directories(self):
        """Create the ~/.datus directory structure"""
        print("🚀 Setting up Datus Agent user directories...")
        print(f"📁 Creating directories in {self.datus_dir}")

        for directory in [self.data_dir, self.conf_dir, self.template_dir, self.sample_dir]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Created: {directory}")
                logger.info(f"Created directory: {directory}")
            else:
                print(f"   ℹ️  Already exists: {directory}")
                logger.info(f"Directory already exists: {directory}")

    def find_package_data(self):
        """Find package data files from various possible locations"""
        try:
            import datus

            package_root = Path(datus.__file__).parent.parent

            print("🔍 Looking for data files...")
            print(f"   Package root: {package_root}")

            # Try different possible locations for data files
            possible_data_dirs = [
                package_root / "datus_data",  # For compiled wheel install
                package_root,  # For source install (conf/ and datus/prompts/prompt_templates/)
                Path(datus.__file__).parent / "data",  # Alternative location
            ]

            for data_dir_candidate in possible_data_dirs:
                print(f"   Checking: {data_dir_candidate}")
                if data_dir_candidate.exists():
                    print(f"   ✅ Found data source: {data_dir_candidate}")
                    logger.info(f"Found data source: {data_dir_candidate}")
                    return data_dir_candidate

            print("   ❌ No data source found")
            logger.warning("No package data source found")
            return None

        except Exception as e:
            print(f"   ❌ Error finding package data: {e}")
            logger.error(f"Error finding package data: {e}")
            return None

    def copy_configuration_files(self, data_source):
        """Copy configuration files from package data to user directory"""
        # For datus_data structure (wheel install)
        source_conf = data_source / "conf"
        if not source_conf.exists():
            # For source install structure
            source_conf = data_source / "conf"

        if not source_conf.exists():
            print(f"   ⚠️  No conf directory found in {data_source}")
            logger.warning(f"No conf directory found in {data_source}")
            return 0

        print("📋 Copying configuration files...")
        conf_count = 0

        for conf_file in source_conf.iterdir():
            if conf_file.is_file():
                target_file = self.conf_dir / conf_file.name
                if not target_file.exists():  # Don't overwrite existing config
                    shutil.copy2(conf_file, target_file)
                    print(f"   ✅ Copied: {conf_file.name}")
                    logger.info(f"Copied config file: {conf_file.name}")
                    conf_count += 1
                else:
                    print(f"   ⏭️  Skipped (exists): {conf_file.name}")
                    logger.info(f"Skipped existing config: {conf_file.name}")

        print(f"   📋 Copied {conf_count} configuration files")
        return conf_count

    def copy_template_files(self, data_source):
        """Copy template files from package data to user directory"""
        # For datus_data structure (wheel install)
        source_template = data_source / "template"
        if not source_template.exists():
            # For source install structure
            source_template = data_source / "datus" / "prompts" / "prompt_templates"

        if not source_template.exists():
            print(f"   ⚠️  No template directory found in {data_source}")
            logger.warning(f"No template directory found in {data_source}")
            return 0

        print("📄 Copying template files...")
        template_count = 0

        for template_file in source_template.rglob("*"):
            if template_file.is_file():
                rel_path = template_file.relative_to(source_template)
                target_file = self.template_dir / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)

                if not target_file.exists():  # Don't overwrite existing files
                    shutil.copy2(template_file, target_file)
                    print(f"   ✅ Copied: {rel_path}")
                    logger.info(f"Copied template file: {rel_path}")
                    template_count += 1
                else:
                    print(f"   ⏭️  Skipped (exists): {rel_path}")
                    logger.info(f"Skipped existing template: {rel_path}")

        print(f"   📄 Copied {template_count} template files")
        return template_count

    def copy_sample_files(self, data_source):
        """Copy sample files from package data to user directory"""
        # For datus_data structure (wheel install)
        source_sample = data_source / "sample"
        if not source_sample.exists():
            # For source install structure
            source_sample = data_source / "tests"

        if not source_sample.exists():
            print(f"   ⚠️  No sample directory found in {data_source}")
            logger.warning(f"No sample directory found in {data_source}")
            return 0

        print("🗂️  Copying sample files...")
        sample_count = 0

        # Handle different source types
        if source_sample.name == "tests":
            # For source install, specifically look for duckdb-demo.duckdb
            demo_file = source_sample / "duckdb-demo.duckdb"
            if demo_file.exists():
                target_file = self.sample_dir / demo_file.name
                if not target_file.exists():  # Don't overwrite existing files
                    shutil.copy2(demo_file, target_file)
                    print(f"   ✅ Copied: {demo_file.name}")
                    logger.info(f"Copied sample file: {demo_file.name}")
                    sample_count += 1
                else:
                    print(f"   ⏭️  Skipped (exists): {demo_file.name}")
                    logger.info(f"Skipped existing sample: {demo_file.name}")
        else:
            # For wheel install (sample directory), copy all files
            for sample_file in source_sample.rglob("*"):
                if sample_file.is_file():
                    rel_path = sample_file.relative_to(source_sample)
                    target_file = self.sample_dir / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)

                    if not target_file.exists():  # Don't overwrite existing files
                        shutil.copy2(sample_file, target_file)
                        print(f"   ✅ Copied: {rel_path}")
                        logger.info(f"Copied sample file: {rel_path}")
                        sample_count += 1
                    else:
                        print(f"   ⏭️  Skipped (exists): {rel_path}")
                        logger.info(f"Skipped existing sample: {rel_path}")

        print(f"   🗂️  Copied {sample_count} sample files")
        return sample_count

    def initialize(self):
        """Main initialization method"""
        print("Datus Agent Initialization")
        print("=" * 40)

        try:
            # Create directories
            self.create_directories()

            # Find and copy data files
            data_source = self.find_package_data()

            if data_source:
                conf_count = self.copy_configuration_files(data_source)
                template_count = self.copy_template_files(data_source)
                sample_count = self.copy_sample_files(data_source)

                print("✅ Initialization complete!")
                print(f"   📋 {conf_count} configuration files copied")
                print(f"   📄 {template_count} template files copied")
                print(f"   🗂️  {sample_count} sample files copied")

                print(f"\n📍 Datus directories created at: {self.datus_dir}")
                print("\n🎉 Next steps:")
                print("   1. Configure your databases in ~/.datus/conf/")
                print("   2. Check sample files in ~/.datus/sample/")
                print("   3. Run 'datus --help' or 'datus-cli --help' to get started")

                logger.info("Datus initialization completed successfully")
                return True

            else:
                print("\n⚠️  Initialization completed with warnings:")
                print("   📁 Directories created, but no data files copied")
                print("   💡 This might be normal for wheel installations")

                print(f"\n📍 Datus directories created at: {self.datus_dir}")
                print("\n🎉 Next steps:")
                print("   1. Configure your databases in ~/.datus/conf/")
                print("   2. Check sample files in ~/.datus/sample/")
                print("   3. Run 'datus --help' or 'datus-cli --help' to get started")

                logger.info("Datus initialization completed successfully")
                return True

        except Exception as e:
            print(f"\n❌ Initialization failed: {e}")
            print("\nYou can still use Datus Agent, but you may need to:")
            print("   1. Manually create ~/.datus/data, ~/.datus/conf, ~/.datus/template, ~/.datus/sample directories")
            print("   2. Copy configuration and sample files manually if needed")

            logger.error(f"Initialization failed: {e}")
            return False


def main():
    """Entry point for the datus-init command"""
    try:
        initializer = DatusInitializer()
        success = initializer.initialize()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n❌ Initialization cancelled by user")
        logger.info("Initialization cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Unexpected error during initialization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
