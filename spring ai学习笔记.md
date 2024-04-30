# spring ai学习笔记

## 概念速览

## 最速上手

### 初体验

#### 准备工作

至少安装jdk17

创建springboot项目，starter里选择spring ai。你使用哪一个就勾选哪一个。

在application.yml文件里写上必要的配置。例如如果是使用别人的大模型，就需要配api-key。还有指定使用哪一个大模型。

#### 开始编码

只要是引入了对应的starter，配置文件里写了必要的配置，chatClient的bean就已经存在于spring容器中，直接注入进来调用call方法。

```java
@RestController
public class SimpleAiController {

	private final ChatClient chatClient;

	@Autowired
	public SimpleAiController(@Qualifier("azureOpenAiChatClient") ChatClient chatClient) {
		this.chatClient = chatClient;
	}

	@GetMapping("/ai/simple")
	public Map<String, String> generation(
			@RequestParam(value = "message", defaultValue = "一出门就被丘丘人盯上里怎么办") String message) {
    // 直接调用call方法就能开始对话
		return Map.of("generation", chatClient.call(message));
	}
}
```

### PromptTemplate

在和大模型对话时，可以使用模版。

模版简单理解为，一个字符串，里面有占位符。

例如：

```st
有位音乐人叫{musician}，有首歌叫{songName}，帮我介绍一下这首歌的创作思路
```

```java
@RestController
public class PromptTemplateController {

	private final ChatClient chatClient;

	@Value("classpath:/prompts/prompt.st")
	private Resource resource;

	@Autowired
	public PromptTemplateController(ChatClient chatClient) {
		this.chatClient = chatClient;
	}

	@GetMapping("/ai/prompt")
	public ChatResponse completion(@RequestParam(value = "musician", defaultValue = "skrillex") String musician,
			@RequestParam(value = "songName", defaultValue = "Rumble") String songName) {
    // 1、创建一个提示模版
		var promptTemplate = new PromptTemplate(resource);
    // 2、根据模版生成一个具体的提示
		var prompt = promptTemplate.create(Map.of("musician", musician, "songName", songName));
    // 3、调用
		return chatClient.call(prompt);
	}
}
```

### Role

提示工程可是一门大学问，如何让大模型回答的更好，给他设置一个角色，让他知道他是谁。

```st
你是一个有用的AI助手。
你是原神游戏里旅行者的提瓦特大陆向导。
你的名字是{name}
你应该以{name}的名义和{voice}的风格回答用户的请求。
```

```java
@RestController
public class RoleController {

	private final ChatClient chatClient;

	@Value("classpath:/prompts/system-message.st")
	private Resource systemResource;

	@Autowired
	public RoleController(ChatClient chatClient) {
		this.chatClient = chatClient;
	}

	@GetMapping("/ai/roles")
	public ChatResponse generate(@RequestParam(value = "message",
			defaultValue = "从蒙徳到璃月怎么走") String message,
															 @RequestParam(value = "name", defaultValue = "派蒙") String name,
															 @RequestParam(value = "voice", defaultValue = "二次元美少女") String voice) {
		// 1、创建用户消息
    var userMessage = new UserMessage(message);
    // 2、创建系统消息模版
		var systemPromptTemplate = new SystemPromptTemplate(systemResource);
    // 3、根据系统消息模版生成具体的系统消息
		var systemMessage = systemPromptTemplate.createMessage(Map.of("name", name, "voice", voice));
    // 4、组合成一个prompt
		var prompt = new Prompt(List.of(userMessage, systemMessage));
    // 5、调用
		return chatClient.call(prompt);
	}
}
```

### outputParser

想让大模型更好的被我们所使用，只返回文本太局限了。让他直接返回给我们标准的json或者是map，或者直接映射到对象里。

```st
给我介绍一下手游三国杀里的武将：{sgsRoleName}
{format}
```

```java
public record SgsRoleCard(String name, Integer health, List<Skill> skills) {
}

record Skill(String name, String description) {
}
```

```java
@RestController
public class OutputParserController {

	private final ChatClient chatClient;

	@Value("classpath:/prompts/output-parser-message.st")
	private Resource resource;

	@Autowired
	public OutputParserController(ChatClient chatClient) {
		this.chatClient = chatClient;
	}

	@GetMapping("/ai/output")
	public SgsRoleCard generate(@RequestParam(value = "sgsRoleName", defaultValue = "界徐盛") String sgsRoleName) {
    // 1、创建一个输出解析器，这里选择解析为bean（还可以选择为json或者map）
		var outputParser = new BeanOutputParser<>(SgsRoleCard.class);
    // 2、创建一个模版，把解析器里的format也传进去
		var promptTemplate = new PromptTemplate(resource, Map.of("sgsRoleName", sgsRoleName, "format", outputParser.getFormat()));
    // 3、创建具体的prompt
		var prompt = promptTemplate.create();
    // 4、调用call方法并进行解析
		return outputParser.parse(chatClient.call(prompt).getResult().getOutput().getContent());
	}
}
```

