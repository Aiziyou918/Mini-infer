#include <gtest/gtest.h>
#include "mini_infer/runtime/optimization_profile.h"

using namespace mini_infer;
using namespace mini_infer::runtime;

// ============================================================================
// ShapeRange Tests
// ============================================================================

TEST(ShapeRangeTest, ValidRange) {
    ShapeRange range(
        core::Shape({1, 3, 224, 224}),   // min
        core::Shape({4, 3, 384, 384}),   // opt
        core::Shape({8, 3, 512, 512})    // max
    );
    
    EXPECT_TRUE(range.is_valid());
}

TEST(ShapeRangeTest, InvalidRangeDifferentNdim) {
    ShapeRange range(
        core::Shape({1, 3, 224, 224}),   // 4D
        core::Shape({4, 3, 384}),        // 3D - invalid!
        core::Shape({8, 3, 512, 512})    // 4D
    );
    
    EXPECT_FALSE(range.is_valid());
}

TEST(ShapeRangeTest, InvalidRangeMinGreaterThanOpt) {
    ShapeRange range(
        core::Shape({8, 3, 224, 224}),   // min too large
        core::Shape({4, 3, 384, 384}),   // opt
        core::Shape({16, 3, 512, 512})   // max
    );
    
    EXPECT_FALSE(range.is_valid());
}

TEST(ShapeRangeTest, InvalidRangeOptGreaterThanMax) {
    ShapeRange range(
        core::Shape({1, 3, 224, 224}),   // min
        core::Shape({16, 3, 512, 512}),  // opt too large
        core::Shape({8, 3, 384, 384})    // max
    );
    
    EXPECT_FALSE(range.is_valid());
}

TEST(ShapeRangeTest, ContainsValidShape) {
    ShapeRange range(
        core::Shape({1, 3, 224, 224}),
        core::Shape({4, 3, 384, 384}),
        core::Shape({8, 3, 512, 512})
    );
    
    // Within range
    EXPECT_TRUE(range.contains(core::Shape({1, 3, 224, 224})));  // min
    EXPECT_TRUE(range.contains(core::Shape({4, 3, 384, 384})));  // opt
    EXPECT_TRUE(range.contains(core::Shape({8, 3, 512, 512})));  // max
    EXPECT_TRUE(range.contains(core::Shape({2, 3, 300, 300})));  // between
}

TEST(ShapeRangeTest, ContainsOutOfRangeShape) {
    ShapeRange range(
        core::Shape({1, 3, 224, 224}),
        core::Shape({4, 3, 384, 384}),
        core::Shape({8, 3, 512, 512})
    );
    
    // Out of range
    EXPECT_FALSE(range.contains(core::Shape({16, 3, 224, 224})));   // batch too large
    EXPECT_FALSE(range.contains(core::Shape({1, 3, 1024, 1024})));  // H/W too large
    EXPECT_FALSE(range.contains(core::Shape({0, 3, 224, 224})));    // batch too small
}

TEST(ShapeRangeTest, ContainsWrongNdim) {
    ShapeRange range(
        core::Shape({1, 3, 224, 224}),
        core::Shape({4, 3, 384, 384}),
        core::Shape({8, 3, 512, 512})
    );
    
    // Wrong ndim
    EXPECT_FALSE(range.contains(core::Shape({1, 3, 224})));       // 3D
    EXPECT_FALSE(range.contains(core::Shape({1, 3, 224, 224, 1}))); // 5D
}

TEST(ShapeRangeTest, DynamicDimensions) {
    ShapeRange range(
        core::Shape({-1, 3, 224, 224}),  // dynamic batch
        core::Shape({-1, 3, 384, 384}),
        core::Shape({-1, 3, 512, 512})
    );
    
    // Should be valid (dynamic dims are skipped in validation)
    EXPECT_TRUE(range.is_valid());
    
    // Any batch size should be valid
    EXPECT_TRUE(range.contains(core::Shape({1, 3, 300, 300})));
    EXPECT_TRUE(range.contains(core::Shape({100, 3, 400, 400})));
}

// ============================================================================
// OptimizationProfile Tests
// ============================================================================

TEST(OptimizationProfileTest, SetAndGetShapeRange) {
    OptimizationProfile profile;
    
    auto status = profile.set_shape_range(
        "input",
        core::Shape({1, 3, 224, 224}),
        core::Shape({4, 3, 384, 384}),
        core::Shape({8, 3, 512, 512})
    );
    
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    const auto* range = profile.get_shape_range("input");
    ASSERT_NE(range, nullptr);
    EXPECT_EQ(range->min.to_string(), "[1, 3, 224, 224]");
    EXPECT_EQ(range->opt.to_string(), "[4, 3, 384, 384]");
    EXPECT_EQ(range->max.to_string(), "[8, 3, 512, 512]");
}

TEST(OptimizationProfileTest, SetInvalidShapeRange) {
    OptimizationProfile profile;
    
    // Invalid: min > opt
    auto status = profile.set_shape_range(
        "input",
        core::Shape({8, 3, 224, 224}),   // too large
        core::Shape({4, 3, 384, 384}),
        core::Shape({16, 3, 512, 512})
    );
    
    EXPECT_NE(status, core::Status::SUCCESS);
    
    // Should not be stored
    const auto* range = profile.get_shape_range("input");
    EXPECT_EQ(range, nullptr);
}

TEST(OptimizationProfileTest, GetNonexistentRange) {
    OptimizationProfile profile;
    
    const auto* range = profile.get_shape_range("nonexistent");
    EXPECT_EQ(range, nullptr);
}

TEST(OptimizationProfileTest, IsValidForShapes) {
    OptimizationProfile profile;
    
    profile.set_shape_range(
        "input",
        core::Shape({1, 3, 224, 224}),
        core::Shape({4, 3, 384, 384}),
        core::Shape({8, 3, 512, 512})
    );
    
    // Valid shapes
    EXPECT_TRUE(profile.is_valid_for({{"input", core::Shape({1, 3, 224, 224})}}));
    EXPECT_TRUE(profile.is_valid_for({{"input", core::Shape({4, 3, 384, 384})}}));
    EXPECT_TRUE(profile.is_valid_for({{"input", core::Shape({8, 3, 512, 512})}}));
    EXPECT_TRUE(profile.is_valid_for({{"input", core::Shape({2, 3, 300, 300})}}));
    
    // Invalid shapes
    EXPECT_FALSE(profile.is_valid_for({{"input", core::Shape({16, 3, 224, 224})}}));  // batch too large
    EXPECT_FALSE(profile.is_valid_for({{"input", core::Shape({1, 3, 1024, 1024})}})); // H/W too large
}

TEST(OptimizationProfileTest, IsValidForMultipleInputs) {
    OptimizationProfile profile;
    
    profile.set_shape_range(
        "input1",
        core::Shape({1, 3, 224, 224}),
        core::Shape({4, 3, 384, 384}),
        core::Shape({8, 3, 512, 512})
    );
    
    profile.set_shape_range(
        "input2",
        core::Shape({1, 256}),
        core::Shape({4, 512}),
        core::Shape({8, 1024})
    );
    
    // All inputs valid
    EXPECT_TRUE(profile.is_valid_for({
        {"input1", core::Shape({2, 3, 300, 300})},
        {"input2", core::Shape({2, 400})}
    }));
    
    // One input out of range
    EXPECT_FALSE(profile.is_valid_for({
        {"input1", core::Shape({2, 3, 300, 300})},
        {"input2", core::Shape({2, 2000})}  // too large
    }));
    
    // Missing input
    EXPECT_FALSE(profile.is_valid_for({
        {"input1", core::Shape({2, 3, 300, 300})}
        // input2 missing
    }));
}

TEST(OptimizationProfileTest, GetInputNames) {
    OptimizationProfile profile;
    
    profile.set_shape_range("input1",
        core::Shape({1, 3, 224, 224}),
        core::Shape({4, 3, 384, 384}),
        core::Shape({8, 3, 512, 512})
    );
    
    profile.set_shape_range("input2",
        core::Shape({1, 256}),
        core::Shape({4, 512}),
        core::Shape({8, 1024})
    );
    
    auto names = profile.get_input_names();
    EXPECT_EQ(names.size(), 2);
    EXPECT_TRUE(std::find(names.begin(), names.end(), "input1") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "input2") != names.end());
}

TEST(OptimizationProfileTest, GetOptimalShapes) {
    OptimizationProfile profile;
    
    profile.set_shape_range("input",
        core::Shape({1, 3, 224, 224}),
        core::Shape({4, 3, 384, 384}),   // optimal
        core::Shape({8, 3, 512, 512})
    );
    
    auto optimal = profile.get_optimal_shapes();
    ASSERT_EQ(optimal.size(), 1);
    EXPECT_EQ(optimal["input"].to_string(), "[4, 3, 384, 384]");
}

TEST(OptimizationProfileTest, EmptyProfile) {
    OptimizationProfile profile;
    
    EXPECT_TRUE(profile.empty());
    EXPECT_EQ(profile.size(), 0);
    
    profile.set_shape_range("input",
        core::Shape({1, 3, 224, 224}),
        core::Shape({4, 3, 384, 384}),
        core::Shape({8, 3, 512, 512})
    );
    
    EXPECT_FALSE(profile.empty());
    EXPECT_EQ(profile.size(), 1);
}

TEST(OptimizationProfileTest, ClearProfile) {
    OptimizationProfile profile;
    
    profile.set_shape_range("input",
        core::Shape({1, 3, 224, 224}),
        core::Shape({4, 3, 384, 384}),
        core::Shape({8, 3, 512, 512})
    );
    
    EXPECT_FALSE(profile.empty());
    
    profile.clear();
    
    EXPECT_TRUE(profile.empty());
    EXPECT_EQ(profile.size(), 0);
    EXPECT_EQ(profile.get_shape_range("input"), nullptr);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


