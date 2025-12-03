#pragma once

#include <stdint.h>

#include <string>
#include <tuple>

constexpr size_t TOK2INDEX_SIZE = 96;

extern std::tuple<const char *, uint16_t> _Tok2Index[TOK2INDEX_SIZE];