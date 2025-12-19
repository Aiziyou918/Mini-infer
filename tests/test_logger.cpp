#include <gtest/gtest.h>

#include <sstream>

#include "mini_infer/utils/logger.h"


using namespace mini_infer;

class LoggerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        utils::Logger::set_level(utils::LogLevel::DEBUG);
    }

    void TearDown() override {
        utils::Logger::set_level(utils::LogLevel::INFO);
    }
};

TEST_F(LoggerTest, SetAndGetLevel) {
    utils::Logger::set_level(utils::LogLevel::WARNING);
    EXPECT_EQ(utils::Logger::get_level(), utils::LogLevel::WARNING);

    utils::Logger::set_level(utils::LogLevel::ERROR);
    EXPECT_EQ(utils::Logger::get_level(), utils::LogLevel::ERROR);

    utils::Logger::set_level(utils::LogLevel::INFO);
    EXPECT_EQ(utils::Logger::get_level(), utils::LogLevel::INFO);
}

TEST_F(LoggerTest, DebugLevel) {
    utils::Logger::set_level(utils::LogLevel::DEBUG);
    EXPECT_EQ(utils::Logger::get_level(), utils::LogLevel::DEBUG);
}

TEST_F(LoggerTest, InfoLevel) {
    utils::Logger::set_level(utils::LogLevel::INFO);
    EXPECT_EQ(utils::Logger::get_level(), utils::LogLevel::INFO);
}

TEST_F(LoggerTest, WarningLevel) {
    utils::Logger::set_level(utils::LogLevel::WARNING);
    EXPECT_EQ(utils::Logger::get_level(), utils::LogLevel::WARNING);
}

TEST_F(LoggerTest, ErrorLevel) {
    utils::Logger::set_level(utils::LogLevel::ERROR);
    EXPECT_EQ(utils::Logger::get_level(), utils::LogLevel::ERROR);
}

TEST_F(LoggerTest, LevelHierarchy) {
    utils::Logger::set_level(utils::LogLevel::DEBUG);
    EXPECT_EQ(utils::Logger::get_level(), utils::LogLevel::DEBUG);

    utils::Logger::set_level(utils::LogLevel::INFO);
    EXPECT_NE(utils::Logger::get_level(), utils::LogLevel::DEBUG);

    utils::Logger::set_level(utils::LogLevel::WARNING);
    EXPECT_NE(utils::Logger::get_level(), utils::LogLevel::INFO);

    utils::Logger::set_level(utils::LogLevel::ERROR);
    EXPECT_NE(utils::Logger::get_level(), utils::LogLevel::WARNING);
}

TEST_F(LoggerTest, DefaultLevel) {
    utils::Logger::set_level(utils::LogLevel::INFO);
    EXPECT_EQ(utils::Logger::get_level(), utils::LogLevel::INFO);
}

TEST_F(LoggerTest, MultipleSetLevel) {
    for (int i = 0; i < 10; ++i) {
        utils::Logger::set_level(utils::LogLevel::DEBUG);
        EXPECT_EQ(utils::Logger::get_level(), utils::LogLevel::DEBUG);

        utils::Logger::set_level(utils::LogLevel::ERROR);
        EXPECT_EQ(utils::Logger::get_level(), utils::LogLevel::ERROR);
    }
}

TEST_F(LoggerTest, LevelPersistence) {
    utils::Logger::set_level(utils::LogLevel::WARNING);

    auto level1 = utils::Logger::get_level();
    auto level2 = utils::Logger::get_level();

    EXPECT_EQ(level1, level2);
    EXPECT_EQ(level1, utils::LogLevel::WARNING);
}
